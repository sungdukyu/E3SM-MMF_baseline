import os, copy, ngl, glob, xarray as xr, numpy as np
class clr:END,RED,GREEN,MAGENTA,CYAN,BOLD = '\033[0m','\033[31m','\033[32m','\033[35m','\033[36m','\033[1m'
#-------------------------------------------------------------------------------
# a usefule conda env for running this script:
# conda create --name pyn_env --channel conda-forge pyngl xarray dask netcdf4
#-------------------------------------------------------------------------------
input_data_root = './metrics_netcdf'

fig_type = 'png'

model_list = []
model_list.append('CNN')
model_list.append('cVAE')
model_list.append('ED')
model_list.append('HSR')
model_list.append('MLP')
model_list.append('RPN')

metric_list = []
# metric_list.append('MAE')
metric_list.append('R2')
# metric_list.append('RMSE')

var_list = []
var_list.append('ptend_t')
var_list.append('ptend_q0001')
var_list.append('cam_out_NETSW')
var_list.append('cam_out_FLWDS')
var_list.append('cam_out_PRECSC')
var_list.append('cam_out_PRECC')
var_list.append('cam_out_SOLS')
var_list.append('cam_out_SOLL')
var_list.append('cam_out_SOLSD')
var_list.append('cam_out_SOLLD')

# sepcify which level to use if applicable
k_lev = 35 # 35 ~> 500mb

num_plot_col = 2

# scrip file for native grid plot
scrip_file_name = 'ne4pg2_scrip.nc'

#---------------------------------------------------------------------------------------------------
# routine for printing various helpful statistics
def print_stat(x,fmt='f',stat='naxh',indent='',compact=False):
   """ Print min, avg, max, and std deviation of input """
   if fmt=='f' : fmt = '%.4f'
   if fmt=='e' : fmt = '%e'
   msg = ''
   line = f'{indent}'
   if not compact: msg += line+'\n'
   for c in list(stat):
      if not compact: line = indent
      if c=='h': line += 'shp: '+str(x.shape)
      if c=='a': line += 'avg: '+fmt%x.mean()
      if c=='n': line += 'min: '+fmt%x.min()
      if c=='x': line += 'max: '+fmt%x.max()
      if c=='s': line += 'std: '+fmt%x.std()
      if not compact: msg += line+'\n'
      if compact: line += ' '*2
   if compact: msg += line#+'\n'
   # print(msg)
   return msg
#---------------------------------------------------------------------------------------------------
# set up the plot resources
res = ngl.Resources()
res.nglDraw,res.nglFrame         = False, False # turn off the automatic drawing/framing
res.tmXBOn,res.tmYLOn            = False, False # Turn off tick marks
# modify the color bar (i.e. "labelbar")
res.lbLabelBarOn                 = True
res.lbLabelFontHeightF           = 0.014
res.lbOrientation                = 'Horizontal'
res.lbLabelFontHeightF           = 0.008
# switch from contrours to filled cells
res.cnLinesOn                    = False
res.cnLineLabelsOn               = False
res.cnInfoLabelOn                = False
res.cnFillOn                     = True
# turn off color bar and use a single common colorbar
res.lbLabelBarOn                 = False
# turn off map grid and center on the Pacific
res.mpGridAndLimbOn              = False
res.mpCenterLonF                 = 180
# Set up cell fill attributes using scrip grid file
scripfile = xr.open_dataset(scrip_file_name).rename({'grid_size':'ncol'})
res.cnFillMode    = 'CellFill'
res.sfXArray      = scripfile['grid_center_lon'].values
res.sfYArray      = scripfile['grid_center_lat'].values
res.sfXCellBounds = scripfile['grid_corner_lon'].values
res.sfYCellBounds = scripfile['grid_corner_lat'].values
### use this to zoom in on a region
# res.mpLimitMode = 'LatLon' 
# res.mpMinLatF   = 45 -15
# res.mpMaxLatF   = 45 +15
# res.mpMinLonF   = 180-15
# res.mpMaxLonF   = 180+15
#---------------------------------------------------------------------------------------------------
# routine to add subtitles to the top of plot
def set_subtitles(wks, plot, left_string='', center_string='', right_string='', font_height=0.01):
   ttres          = ngl.Resources()
   ttres.nglDraw  = False
   # Use plot extent to call ngl.text(), otherwise you will see this error: 
   # GKS ERROR NUMBER   51 ISSUED FROM SUBROUTINE GSVP  : --RECTANGLE DEFINITION IS INVALID
   strx,stry = ngl.get_float(plot,'trXMinF'),ngl.get_float(plot,'trYMinF')
   ttres.txFontHeightF = font_height
   # Set annotation resources to describe how close text is to be attached to plot
   amres = ngl.Resources()
   amres.amOrthogonalPosF = -0.52   # Top of plot plus extra to stay off the border
   if hasattr(ttres,'amOrthogonalPosF'): amres.amOrthogonalPosF = ttres.amOrthogonalPosF
   # Add left string
   amres.amJust,amres.amParallelPosF = 'BottomLeft', -0.5   # Left-justified
   tx_id_l   = ngl.text(wks, plot, left_string, strx, stry, ttres)
   anno_id_l = ngl.add_annotation(plot, tx_id_l, amres)
   # Add center string
   tx_id_c = ngl.text(wks, plot, center_string, strx, stry, ttres)
   amres.amJust,amres.amParallelPosF = "BottomCenter", 0.0   # Centered
   anno_id_c = ngl.add_annotation(plot, tx_id_c, amres)
   # Add right string
   amres.amJust,amres.amParallelPosF = 'BottomRight', 0.5   # Right-justified
   tx_id_r   = ngl.text(wks, plot, right_string, strx, stry, ttres)
   anno_id_r = ngl.add_annotation(plot, tx_id_r, amres)
   return
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
for v in range(len(var_list)):
   print('-'*80)
   print(' '*2+f'var: {clr.MAGENTA}{var_list[v]}{clr.END}\n')
   data_list = [] # this list is just a convenient way to make sure color bars are consistent
   for j in range(len(metric_list)):
      print(' '*4+f'metric: {clr.GREEN}{metric_list[j]}{clr.END}')
      #-------------------------------------------------------------------------
      # output figure file name and type
      fig_file = f'metrics.map.{var_list[v]}.{metric_list[j]}'
      # create plot workstation and plot object
      wks = ngl.open_wks(fig_type,fig_file)
      plot = [None]*len(model_list)
      #-------------------------------------------------------------------------
      for i in range(len(model_list)):
         # print(' '*6+f'model: {model_list[i]}')
         #----------------------------------------------------------------------
         input_file_path = f'{input_data_root}/{model_list[i]}_{metric_list[j]}.nc'
         ds = xr.open_dataset( input_file_path )
         data = ds[var_list[v]]
         if 'lev' in data.dims : data = data.isel(lev=k_lev)
         data_list.append( data.values )
         #----------------------------------------------------------------------
         stat_msg = print_stat(data,stat='naxh',compact=True,indent=' '*8)
         print(' '*6+f'model: {model_list[i]:6} {stat_msg}')
      #-------------------------------------------------------------------------
      # get min and max so that colorbar levels are consistent
      data_min = np.nanmin([np.nanmin(np.ma.masked_invalid(d)) for d in data_list])
      data_max = np.nanmax([np.nanmax(np.ma.masked_invalid(d)) for d in data_list])
      if metric_list[j]=='R2': data_min,data_max = 0.,1.
      print(' '*6+f'data min/max: {data_min} / {data_max}')
      #-------------------------------------------------------------------------
      aboutZero = False
      clev_tup = ngl.nice_cntr_levels(data_min, data_max,cint=None, max_steps=21,aboutZero=aboutZero )
      cmin,cmax,cint = clev_tup
      for i in range(len(model_list)):
         tres = copy.deepcopy(res)
         tres.cnFillPalette = 'MPL_viridis'
         tres.cnLevels = np.linspace(cmin,cmax,num=21)
         tres.cnLevelSelectionMode = 'ExplicitLevels'
         #----------------------------------------------------------------------
         plot[i] = ngl.contour_map(wks,np.ma.masked_invalid(data_list[i]),tres) 
         set_subtitles( wks, plot[i], font_height=0.015,
                        left_string=var_list[v], 
                        center_string=metric_list[j], 
                        right_string=model_list[i])
      #-------------------------------------------------------------------------
      # Finalize plot
      layout = [int(np.ceil(len(plot)/float(num_plot_col))),num_plot_col]
      pnl_res = ngl.Resources()
      pnl_res.nglPanelYWhiteSpacePercent = 5
      pnl_res.nglPanelLabelBar = True
      # draw and frame the plots
      ngl.panel(wks,plot,layout,pnl_res); ngl.destroy(wks)
      # trim the white space using imagemagik
      os.system(f'convert -trim +repage {fig_file}.{fig_type}   {fig_file}.{fig_type}')
      print(' '*4+f'figure: {clr.BOLD}{fig_file}.{fig_type}{clr.END}\n')
      #-------------------------------------------------------------------------
# print another line
print('-'*80)
ngl.end()
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------