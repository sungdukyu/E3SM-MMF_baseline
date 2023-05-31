# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
PATH="$HOME/.local/bin:$HOME/bin:$PATH"
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

alias project='cd /ocean/projects/atm200007p/jlin96'
alias q='sinfo -o "%20P %5a %.10l %16F"'
alias jobs='sacct --format="JobID,State,JobName%30,Elapsed,Timelimit,NNodes,Partition,Start,nodelist" | grep -v CANCELLED | grep -v TIMEOUT | grep -v FAILED | grep -v COMPLETED'
alias checkstart='squeue --user=jlin96 --start'

# load conda env
source /opt/packages/anaconda3/etc/profile.d/conda.sh
module load anaconda3

rmshared() {
    interact -n "$1" -p RM-shared -t "$2"
}

RM() {
    interact -p RM -t "$1"
}

ramslam(){
    interact -p RM-512 -t 2:00:00 -N 1
}

interactDGX2(){
    interact -p GPU --gres=gpu:v100-32:16 -t "$1":00:00 
}

gpusmol() {
    interact -p GPU-shared --gres=gpu:v100-16:4 -N "$1"
}

gpubig() {
    interact -p GPU-shared --gres=gpu:v100-32:8 -N "$1" "$2"
}

quickanddirty() {
    interact -n 10 -p RM-shared -t 60:00
}

welp() {
    interact -n 10 -p RM-shared -t 4:00:00
}

tunnel2compute() {
    ssh -L localhost:"$1":localhost:"$1" "$2"
}

jupyternotebookTuning() {
    conda activate tuner_
    jupyter notebook --port="$1" --no-browser --ip=0.0.0.0
}

jnpreprocessing() {
    conda activate preprocessing
    jupyter lab --port="$1" --no-browser --NotebookApp.allow_origin='*' --ip=0.0.0.0
}

jnvisualizing() {
    conda activate visualizing
    jupyter lab --port="$1" --no-browser --NotebookApp.allow_origin='*' --ip=0.0.0.0
}

jnanalysis() {
    conda activate analysis
    jupyter lab --port="$1" --no-browser --ip=0.0.0.0
}

jntf() {
    conda activate tf2
    jupyter lab --port="$1" --no-browser --NotebookApp.allow_origin='*' --ip=0.0.0.0
}

jnleapexample() {
    conda activate leapexample
    jupyter lab --port="$1" --no-browser --NotebookApp.allow_origin='*' --ip=0.0.0.0
}

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/packages/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/packages/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/packages/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/packages/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

