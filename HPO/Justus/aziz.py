import numpy as np
import pickle
from matplotlib import pyplot as plt


def rs(test_y, pred_y):
    """
    test_y: stride-6 val np
    pred_y: stride-6 predictions np
    """
    plt.close('all')

    with open('data/indextolatlons.pkl', 'rb') as f:
        data = pickle.load(f)

    model_name_l = ['cVAE']
    n_model = len(model_name_l)

    lats_unique = []
    lats_unique.append(data[0][0])
    long_unique = []
    long_unique.append(data[0][1])

    for i in range(len(data) - 1):
        if not (data[i + 1][0] in lats_unique):
            lats_unique.append(data[i + 1][0])
        if not (data[i + 1][1] in long_unique):
            long_unique.append(data[i + 1][1])

    Nlats = len(lats_unique)  # 87
    Nlong = len(long_unique)  # 199

    ind_lat = []
    for i in range(Nlats):
        ind_lat_loc = []
        for j in range(len(data)):
            if data[j][0] == lats_unique[i]:
                ind_lat_loc.append(j)
        ind_lat.append(ind_lat_loc)

    lats_unique = np.array(lats_unique)
    arg_lats_unique = np.argsort(lats_unique)
    lats_unique = lats_unique[arg_lats_unique]

    test_y = np.array(test_y, dtype=np.float64)

    pred_y_l = []

    import h5py

    pred_y = np.array(pred_y)
    pred_y_l.append(pred_y)


    def reshape(std_test_H):
        std_test_H_heat = std_test_H[:, :60].reshape((int(std_test_H[:, :60].shape[0] / 384), 384, 60))

        std_test_H_heat_daily = np.mean(std_test_H_heat.reshape((std_test_H_heat.shape[0] // 12, 12, 384, 60)),
                                        axis=1)  # Nday x 384 x 60
        std_test_H_moist = std_test_H[:, 60:120].reshape((int(std_test_H[:, 60:120].shape[0] / 384), 384, 60))
        std_test_H_moist_daily = np.mean(std_test_H_moist.reshape((std_test_H_moist.shape[0] // 12, 12, 384, 60)),
                                         axis=1)  # Nday x 384 x 60

        std_test_H_heat_daily_long = []
        std_test_H_moist_daily_long = []
        for i in range(Nlats):
            std_test_H_heat_daily_long.append(np.mean(std_test_H_heat_daily[:, ind_lat[i], :], axis=1))
            std_test_H_moist_daily_long.append(np.mean(std_test_H_moist_daily[:, ind_lat[i], :], axis=1))
        std_test_H_heat_daily_long = np.array(std_test_H_heat_daily_long)  # lat x Nday x 60
        std_test_H_moist_daily_long = np.array(std_test_H_moist_daily_long)  # lat x Nday x 60

        return std_test_H_heat_daily_long, std_test_H_moist_daily_long


    test_heat_daily_long, test_moist_daily_long = reshape(test_y)

    # may need to adjust figure size if number of models considered is different from 3
    fig, ax = plt.subplots(2, n_model, figsize=(n_model * 12, 18), squeeze=False)
    y = np.load('data/pressures_val_stride6_60lvls.npy') / 100
    X, Y = np.meshgrid(lats_unique, y)

    for i in range(n_model):
        pred_heat_daily_long, pred_moist_daily_long = reshape(pred_y_l[i])

        coeff = 1 - np.sum((pred_heat_daily_long - test_heat_daily_long) ** 2, axis=1) / np.sum(
            (test_heat_daily_long - np.mean(test_heat_daily_long, axis=1)[:, None, :]) ** 2, axis=1)
        coeff = coeff[arg_lats_unique, :]
        coeff = coeff.T

        contour_plot = ax[0, i].pcolor(X, Y, coeff, cmap='Blues', vmin=0, vmax=1)  # pcolormesh
        ax[0, i].contour(X, Y, coeff, [0.7], colors='pink', linewidths=[4])
        ax[0, i].contour(X, Y, coeff, [0.9], colors='orange', linewidths=[4])
        ax[0, i].set_ylim(ax[0, i].get_ylim()[::-1])
        ax[0, i].set_title(model_name_l[i] + " - Heating")
        ax[0, i].set_xticks([])

        coeff = 1 - np.sum((pred_moist_daily_long - test_moist_daily_long) ** 2, axis=1) / np.sum(
            (test_moist_daily_long - np.mean(test_moist_daily_long, axis=1)[:, None, :]) ** 2, axis=1)
        coeff = coeff[arg_lats_unique, :]
        coeff = coeff.T

        contour_plot = ax[1, i].pcolor(X, Y, coeff, cmap='Blues', vmin=0, vmax=1)  # pcolormesh
        ax[1, i].contour(X, Y, coeff, [0.7], colors='pink', linewidths=[4])
        ax[1, i].contour(X, Y, coeff, [0.9], colors='orange', linewidths=[4])
        ax[1, i].set_ylim(ax[1, i].get_ylim()[::-1])
        ax[1, i].set_title(model_name_l[i] + " - Moistening")

        if i != 0:
            ax[0, i].set_yticks([])
            ax[1, i].set_yticks([])

    # lines below for x and y label axes are valid if 3 models are considered
    # we want to put only one label for each axis
    # if nbr of models is different from 3 please adjust label location to center it
    ax[1, 0].set_xlabel("Degrees Latitude")
    # ax[1,1].xaxis.set_label_coords(-0.10,-0.10)

    ax[0, 0].set_ylabel("Pressure (hPa)")
    ax[0, 0].yaxis.set_label_coords(-0.2, -0.09)  # (-1.38,-0.09)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.12, 0.02, 0.76])
    fig.colorbar(contour_plot, label="Skill Score " + r'$\left(\mathrm{R^{2}}\right)$', cax=cbar_ax)
    plt.suptitle("Baseline models Skill for Vertically Resolved Tendencies", y=0.97)
    plt.subplots_adjust(hspace=0.13)
    plt.show()


def skill(test_y, pred_y):
    """
    test_y: stride-6 val np
    pred_y: stride-6 predictions np
    """
    plt.close('all')

    ######################################
    ############ User's input ############
    ######################################

    # please provide your model_name. The latter will be used in saving the results
    # files (npy and plots)
    model_name_l = ['cVAE']

    # load model target values of shape Npoints x 128
    # pred_y = np.load('pred_y.npy')

    ######################################
    ############# Test steup #############
    ######################################

    mlo_scale = np.array(np.load('data/mlo_scale.npy'), dtype=np.float32)

    # Please specify the temporal stride used in generating validation/test dataset
    # and original GCM time-step (dt_GCM) in minutes
    stride = 6
    dt_GCM = 20

    # load true target values of shape Ndata x 128
    # test_y = np.load('/ocean/projects/atm200007p/jlin96/neurips_proj/e3sm_train_npy/val_target_stride6.npy')


    test_y = test_y / mlo_scale

    pred_y_l = []
    pred_y_l.append(pred_y[:] / mlo_scale)

    # file 'pressures_val_stride6_60lvls.npy' contains pressure levels in Pa
    pressure = np.load('data/pressures_val_stride6_60lvls.npy') / 100
    N_pressure = pressure.shape[0]  # Number of pressure levels: 60

    # load dictionary of latitude-longitude coordinates of GCM grid on which data is saved
    with open('data/indextolatlons.pkl', 'rb') as f:
        data = pickle.load(f)
    N_lat_long = len(data)  # = 384 There are 384 points in latitude-longitude GCM grid

    mweightpre = np.load('data/ne4.val-stride-6.dp-over-g.npy').T  # 60x384 ==> 384x60
    area = np.load('data/ne4.area.npy')[:, None] / 4 / np.pi  # 384 ==> 384x1

    Lv = 2.26e6
    cp = 1.00464e3
    rho = 997

    weight = np.concatenate((cp * mweightpre * area, Lv * mweightpre * area, area, area, rho * Lv * area,
                             rho * Lv * area, area, area, area, area), axis=1)
    # 384 x 128

    # Since in the current training and validation datasets all first 8 altitude levels
    # from TOA show exactly zeros values for moistening tendency, we can discard the
    # corresponding pressure levels in the moistening tendency plot by setting n_remove_moist = 8
    # If you want to keep all pressure levels in the moistening tendency plot, please set n_remove_moist = 0
    n_remove_moist = 0

    test_y = np.array(test_y, dtype=np.float64)
    if 1 == 1:
        for i in range(1):
            pred_y = pred_y_l[i]
            model_name = model_name_l[i]
            # numpy return erronous errors when summing over "large" number of points with float32 format,
            # so we transform arrays to float64 type before computing coefficient of determination R2
            pred_y = np.array(pred_y, dtype=np.float64)  # Npts(time discret \times long-lat discret) x 128

            dim_y = test_y.shape[1]

            # Ndata is total number of data points in validation / test dataset
            # N_time_steps is the number of time-steps considered in validation / test dataset
            Ndata = test_y.shape[0]
            N_time_steps = Ndata // N_lat_long

            ###########################################
            # Global metrics for 128 output variables #
            ###########################################

            def reshape_npy(y):
                # reshape true data into: N_time_steps x N_lat_long x N_pressure
                return y.reshape((N_time_steps, N_lat_long, dim_y))

            test_daily = weight * reshape_npy(test_y)  # N_time_steps x N_lat_long x N_pressure

            pred_daily = weight * reshape_npy(pred_y)

            MAE = np.mean(np.abs(test_daily - pred_daily), axis=(0, 1))
            MAE_f = []
            MAE_f.append(np.mean(MAE[:N_pressure]))
            MAE_f.append(np.mean(MAE[N_pressure:2 * N_pressure]))
            for i in range(8):
                MAE_f.append(MAE[2 * N_pressure + i])
            MAE_f = np.array(MAE_f)
            print('MAE heat: ', MAE_f[0])

            RMSE = np.sqrt(np.mean((test_daily - pred_daily) ** 2, axis=(0, 1)))
            RMSE_f = []
            RMSE_f.append(np.mean(RMSE[:N_pressure]))
            RMSE_f.append(np.mean(RMSE[N_pressure:2 * N_pressure]))
            for i in range(8):
                RMSE_f.append(RMSE[2 * N_pressure + i])
            RMSE_f = np.array(RMSE_f)

            print(i)
            R2 = 1 - np.sum((pred_daily - test_daily) ** 2, axis=(0, 1)) / np.sum(
                (test_daily - np.mean(test_daily, axis=(0, 1))) ** 2, axis=(0, 1))
            R2_f = []
            R2_f.append(np.mean(R2[:N_pressure]))
            R2_f.append(np.mean(R2[N_pressure + 12:2 * N_pressure]))
            for i in range(8):
                R2_f.append(R2[2 * N_pressure + i])
            R2_f = np.array(R2_f)
    plt.bar(range(10), R2_f)
    plt.show()


def crps(test_y, pred_y):
    """
    test_y: stride-6 val np
    pred_y: stride-6 predictions np
    """

    ######################################
    ############ User's input ############
    ######################################

    # please provide your model_name. The latter will be used in saving the results
    # files (npy and plots)
    model_name_l = ['cVAE']

    ######################################
    ############# Test steup #############
    ######################################

    mlo_scale = np.array(np.load('data/mlo_scale.npy'), dtype=np.float32)

    # Please specify the temporal stride used in generating validation/test dataset
    # and original GCM time-step (dt_GCM) in minutes
    stride = 6
    dt_GCM = 20

    pred_y_l = []
    # [B, L, S] -> [S, B, L]
    pred_y_l.append(np.swapaxes(np.swapaxes(pred_y[:], 0, 2), 1, 2) / mlo_scale)

    # file 'pressures_val_stride6_60lvls.npy' contains pressure levels in Pa
    pressure = np.load('data/pressures_val_stride6_60lvls.npy') / 100
    N_pressure = pressure.shape[0]  # Number of pressure levels: 60

    # load dictionary of latitude-longitude coordinates of GCM grid on which data is saved
    with open('data/indextolatlons.pkl', 'rb') as f:
        data = pickle.load(f)
    N_lat_long = len(data)  # = 384 There are 384 points in latitude-longitude GCM grid

    mweightpre = np.load('data/ne4.val-stride-6.dp-over-g.npy').T  # 60x384 ==> 384x60
    area = np.load('data/ne4.area.npy')[:, None] / 4 / np.pi  # 384 ==> 384x1

    Lv = 2.26e6
    cp = 1.00464e3
    rho = 997

    weight = np.concatenate(
        (cp * mweightpre, Lv * mweightpre, area, area, rho * Lv * area, rho * Lv * area, area, area, area, area),
        axis=1)
    # 384 x 128

    # Since in the current training and validation datasets all first 8 altitude levels
    # from TOA show exactly zeros values for moistening tendency, we can discard the
    # corresponding pressure levels in the moistening tendency plot by setting n_remove_moist = 8
    # If you want to keep all pressure levels in the moistening tendency plot, please set n_remove_moist = 0
    n_remove_moist = 0

    test_y = np.array(test_y, dtype=np.float64)

    for i in range(1):
        pred_y = pred_y_l[i]
        model_name = model_name_l[i]
        # numpy return erronous errors when summing over "large" number of points with float32 format,
        # so we transform arrays to float64 type before computing coefficient of determination R2
        pred_y = np.array(pred_y, dtype=np.float64)  # Npts(time discret \times long-lat discret) x 128

        dim_y = test_y.shape[1]

        # Ndata is total number of data points in validation / test dataset
        # N_time_steps is the number of time-steps considered in validation / test dataset
        Ndata = test_y.shape[0]
        N_time_steps = Ndata // N_lat_long

        ###########################################
        # Global metrics for 128 output variables #
        ###########################################

        def reshape_npy(y):
            # reshape true data into: N_time_steps x N_lat_long x N_pressure
            return y.reshape((N_time_steps, N_lat_long, dim_y))

        test_daily = weight * reshape_npy(test_y)  # N_time_steps x N_lat_long x N_pressure
        test_daily = test_daily.reshape(
            (N_time_steps * N_lat_long, dim_y))  # Npts(time discret \times long-lat discret) x 128

        pred_daily = np.zeros((test_y.shape[0], dim_y, pred_y.shape[0]))
        # batch_size, num_variables, num_samples
        for i in range(pred_y.shape[0]):
            pred_daily[:, :, i] = (weight * reshape_npy(pred_y[i, :, :])).reshape((N_time_steps * N_lat_long, dim_y))

        #    pred_daily = weight * reshape_npy(pred_y)
        #    pred_daily = pred_daily.reshape( (N_time_steps*N_lat_long, dim_y) )

        def crps(outputs, target, weights=None):
            """
            Computes the Continuous Ranked Probability Score (CRPS) between the target and the ecdf for each output variable and then takes a weighted average over them.

            Input
            -----
            outputs - float[B, F, S] samples from the model
            target - float[B, F] ground truth target
            """
            n = outputs.shape[2]
            y_hats = np.sort(outputs, axis=-1)
            # E[Y - y]
            mae = np.abs(target[..., None] - y_hats).mean(axis=(0, -1))
            # E[Y - Y'] ~= sum_i sum_j |Y_i - Y_j| / (2 * n * (n-1))
            diff = y_hats[..., 1:] - y_hats[..., :-1]
            count = np.arange(1, n) * np.arange(n - 1, 0, -1)
            crps = mae - (diff * count).sum(axis=-1).mean(axis=0) / (2 * n * (n - 1))
            return crps  # .average(weights=weights)

        crps_f = crps(pred_daily, test_daily)
        print(crps_f.shape)
        print(np.array(crps_f).shape)
        np.save(model_name + '_CRPS_Mike.npy', crps_f)
