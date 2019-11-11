# icp_boxfit
Fit a 3d box to a point cloud, with options to play with the estimation and plot points.
Goal is to explore the stability of ICP and different formulation variations for this
purpose, using all open-source components.

# usage
```
> python icp_boxfit.py --help
usage: icp_boxfit.py [-h] [-p] [-pp] [-s SIGMA] [-N NUMPTS] [-n NUMREPS] [-r]
                     [-i INFILE] [-o OUTFILE] [-x X] [-y Y] [-z Z] [-Y YAW]
                     [-P PITCH] [-R ROLL] [-W W] [-H H] [-L L]

optional arguments:
  -h, --help              show this help message and exit
  -p, --plotbox           plot soln box
  -pp, --plotpts          plot soln box & ptcloud
  -s SIGMA, --sigma SIGMA
                          sigma of noise to add to ptcloud
  -N NUMPTS, --numpts NUMPTS
                          number of points in synthetic ptcloud
  -n NUMREPS, --numreps NUMREPS
                          number of run repetitions
  -r, --runstats          when set, output stats on the run misfit
  -i INFILE, --infile INFILE
                          optional input filename of csv file with x,y,z ptcloud
  -o OUTFILE, --outfile   OUTFILE
                          optional output filename of csv file of synthetically
                          generated x,y,z ptcloud
  -x X, --x X             synth generated box center x coord
  -y Y, --y Y             synth generated box center y coord
  -z Z, --z Z             synth generated box center z coord
  -Y YAW, --yaw YAW       synth generated box yaw
  -P PITCH, --pitch PITCH synth generated box pitch
  -R ROLL, --roll ROLL    synth generated box roll
  -W W, --w W             synth generated box width
  -H H, --h H             synth generated box height
  -L L, --l L             synth generated box length
```

# installation
```
cd icp_boxfit  # (this repo's dir)
python3 -m venv .venv
source .venv/bin/activate
pip install .
python icp_boxfit.py --help
```

