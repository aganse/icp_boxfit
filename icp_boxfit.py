"""A packaged tool to estimate single 3D bounding box for bounding-box shaped ptcloud,
to explore using ICP method, based on all open-source tools/content.
In typical usage one imports this package and then calls the estimate_bbox() function
from other code; demonstrate() is a standalone cmdline demo app to show how to
implement as well as to test algo updates/variations."""

import argparse

import icp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotationMatrixToEulerAngles(R) :
    """Calculates rotation matrix to euler angles.
    Adapted from snippet by Satya Mallick: 
    https://www.learnopencv.com/rotation-matrix-to-euler-angles
    """
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([z, y, x])


def Rmtx(axis, theta):
    """Generate single-axis rotation matrix given axis and angle.
    Axis specified as [1,0,0] or [0,1,0] or [0,0,1].  Theta in degrees.
    """
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

 
def plot_ptcloud(ptcloud, color='k', ax=None):
    """Plot the generated points that are on the box sides, upon which the
    3D box itself is estimated."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ptcloud[:,0], ptcloud[:,1], ptcloud[:,2], c=color, marker='o')
    return ax


def plot_box(boxpts, ax=None, col='black'):
    """Plot the 3D box specified via 'boxpts' which is an 8x3 ndarray of 
    8 xyz triplets of box vertices."""
    fullbox = np.concatenate((boxpts[:4,:],boxpts[0:1,:],boxpts[4:8,:],boxpts[4:5,:],
                              boxpts[5:6,:],boxpts[1:3,:],boxpts[6:8,:],boxpts[3:4,:]), axis=0)
    ax.plot(fullbox[:,0], fullbox[:,1], fullbox[:,2], c=col)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')



def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    Adapted from snippet by Mateen Ulhaq:
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''

    def set_axes_radius(ax, origin, radius):
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def generate_ptcloud_on_box(x1, x2, y1, y2, z1, z2, N):
    """Generate and plot point cloud uniformly distributed on surface of 3d box."""
    boxsides_x1 = np.stack((np.ones(N)*x1, np.random.uniform(y1,y2,N), np.random.uniform(z1,z2,N)))
    boxsides_x2 = np.stack((np.ones(N)*x2, np.random.uniform(y1,y2,N), np.random.uniform(z1,z2,N)))
    boxsides_y1 = np.stack((np.random.uniform(x1,x2,N), np.ones(N)*y1, np.random.uniform(z1,z2,N)))
    boxsides_y2 = np.stack((np.random.uniform(x1,x2,N), np.ones(N)*y2, np.random.uniform(z1,z2,N)))
    boxsides_z1 = np.stack((np.random.uniform(x1,x2,N), np.random.uniform(y1,y2,N), np.ones(N)*z1))
    boxsides_z2 = np.stack((np.random.uniform(x1,x2,N), np.random.uniform(y1,y2,N), np.ones(N)*z2))
    ptcloud = np.concatenate((boxsides_x1, boxsides_x2, boxsides_y1, boxsides_y2, boxsides_z1, boxsides_z2), axis=1)
    ptcloud = np.transpose(ptcloud)
    return ptcloud


def generate_synth_data(x=2, y=0, z=0, w=2, h=2, l=2, yaw=10, pitch=0, roll=0, noise_sigma=0.01, N=150):
    """Generate point cloud randomly distributed along sides of specified bounding box."""

    # Generate an "input" ptcloud to which we'll want to fit a box.
    # convert from center position + extents to box bounds
    x1, x2, y1, y2, z1, z2 = xyzwhl_to_xyz1xyz2(x,y,z, w,h,l)
    A = generate_ptcloud_on_box(x1, x2, y1, y2, z1, z2, N)
    # Rotate
    Rz = Rmtx([0,0,1], yaw*np.pi/180.0)
    Ry = Rmtx([0,1,0], pitch*np.pi/180.0)
    Rx = Rmtx([1,0,0], roll*np.pi/180.0)
    A = np.dot(Rx, A.T).T
    A = np.dot(Ry, A.T).T
    A = np.dot(Rz, A.T).T
    # Add noise
    A += np.random.randn(N*6, 3) * noise_sigma
    np.random.shuffle(A)
    return A


def xyzwhl_to_xyz1xyz2(x,y,z, w,h,l):
    """Convert from box center-extents spec to diaonally-opposite-corner spec for box."""
    x1 = x - w/2.0
    x2 = x + w/2.0
    y1 = y - l/2.0
    y2 = y + l/2.0
    z1 = z - h/2.0
    z2 = z + h/2.0
    return x1, x2, y1, y2, z1, z2


def estimate_bbox(A, x0=0, y0=0, z0=0, w0=2, h0=2, l0=5, tol=0.000001, solnbox=False, solnpts=False):
    """Estimate best-fitting 3D box to given point cloud.
    params:
        A: Nx3 numpy array of point cloud in x,y,z
        x0, y0, z0, w0, h0, l0: optional scalar values of starting estimates for
            box position and width, depth, height; defaults are 0,0,0,2,2,5
        tol: tolerance value passed to the ICP estimation routine
        solnbox: boolean: if true output 8x3 ndarray of the 8 soln box vertexes (eg for plotting)
        solnpts: boolean: if true output Nx3 ndarray of internal ptcloud fitted to input
    returns:
        x_est, y_est, z_est: estimated position of bbox center
        yaw_est, pitch_est, roll_est: estimated orientation of bbox (degrees)
        w_est, h_est, l_est: estimated width, height, length of bbox
        boxpts: if solnbox was true, an 8x3 ndarray of the 8 soln box vertexes (eg for plotting)
        ptcloud: if solnpts was true, a Nx3 ndarray of internal ptcloud fitted to input, else None
    """

    # Create a box-based ptcloud we'll rotate to fit the input ptcloud
    # B = np.copy(A)  # special case for testing - fit exact copy of data
    N = int(A.shape[0]/6)
    A = A[0:N*6, :]  # if A's original lenght wasn't multiple of 6, lop of last few rows so it is
    x1, x2, y1, y2, z1, z2 = xyzwhl_to_xyz1xyz2(x0, y0, z0, w0, h0, l0)
    B = generate_ptcloud_on_box(x1, x2, y1, y2, z1, z2, N)

    # Rotate/translate ptcloud B to fit ptcloud A
    # T is xform mtx of rotations (cols 0:2) & translations (col 3) to make B closest to A
    T, distances, iterations = icp.icp(B, A, tolerance=tol)
    angles = rotationMatrixToEulerAngles(T[:3, :3].T) * 180/np.pi

    x_est, y_est, z_est = ( T[0,3], T[1,3], T[2,3] )
    yaw_est, pitch_est, roll_est = angles[0], angles[1], angles[2]
    w_est, h_est, l_est = w0, h0, l0  # for now just pinning to input estimate - but for future
                               # there are other ICP variants that additionally estimate scaling

    if solnbox:
        # first create initial estimate box
        boxpts= np.ones((8, 4))
        x1, x2, y1, y2, z1, z2 = xyzwhl_to_xyz1xyz2(x0, y0, z0, w0, h0, l0)
        boxpts[:,0:3] = np.copy(np.array([[x1,y1,z1], [x1,y1,z2], [x1,y2,z2], [x1,y2,z1],
                                     [x2,y1,z1], [x2,y1,z2], [x2,y2,z2], [x2,y2,z1]]))
        # now xform that initial box by the transform that was estimated
        boxpts = np.dot(T, boxpts.T).T
    else:
        boxpts = None

    if solnpts:
        ptcloud = np.ones((N*6, 4))
        ptcloud[:,0:3] = np.copy(B)
        ptcloud = np.dot(T, ptcloud.T).T
    else:
        ptcloud = None

    return x_est, y_est, z_est, yaw_est, pitch_est, roll_est, w_est, h_est, l_est, boxpts, ptcloud


def demonstrate(x=5, y=0, z=0, w=2, h=2, l=5, yaw=10, pitch=0, roll=0, infile=None,
                outfile=None, sigma=0.01, N=150, runstats=False, plotbox=False, plotpts=False):
    """A top-level wrapper demonstrating use of these functions estimating box fit."""

    if infile is not None and outfile is None:
        # infile is csv file with 3 cols:  x, y, z in some arbitrary length unit
        A = np.loadtxt(infile, delimiter=',')
    elif infile is None:
        # for now we don't yet have actual ptcloud input, so generate some
        A = generate_synth_data(x, y, z, w, h, l, yaw, pitch, roll, sigma, N)
        if outfile is not None:
            # output synthetic points to csv file
            np.savetxt(outfile, A, delimiter=",")
    else:
        assert infile is None or outfile is None, \
            "only one of infile and outfile may be specified in a run, not both"

    # Estimate the best-fitting bounding box to the ptcloud
    x_est, y_est, z_est, yaw_est, pitch_est, roll_est, w_est, h_est, l_est, B, C = \
        estimate_bbox(A, solnbox=plotbox, solnpts=plotpts)

    # Print results table
    print('               %7s %7s %7s    %7s %7s %7s    %5s %5s %5s' %
        ('x', 'y', 'z',  'yaw', 'pitch', 'roll',  'w', 'h', 'l') )
    print('ground truth:  %7.2f %7.2f %7.2f    %7.2f %7.2f %7.2f    %5.2f %5.2f %5.2f' %
        (x, y, z, yaw, pitch, roll, w, h, l) )
    print('solution est:  %7.2f %7.2f %7.2f    %7.2f %7.2f %7.2f    %5.2f %5.2f %5.2f' %
        (x_est, y_est, z_est, yaw_est, pitch_est, roll_est, w_est, h_est, l_est) )
    print(' ')

    if runstats:
        print('Run stats:')
        print('  num pts:', N)
        print('  sigma:', sigma)
        print(' ', iterations, 'iterations')
        print('  distances mean/std:', distances.mean(), distances.std())

    if plotbox:
        # Plot original ptcloud
        ax = plot_ptcloud(A, color='red')

        if plotpts:
            # Plot internal ptcloud estimate on top of A to verify the box estimate
            ax = plot_ptcloud(C, color='blue', ax=ax)

        # Plot the solution estimate box
        plot_box(B, ax=ax, col='black')
        set_axes_equal(ax) 
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plotbox", help="plot soln box", action="store_true", default=False)
    parser.add_argument("-pp", "--plotpts", help="plot soln box & ptcloud", action="store_true", default=False)
    parser.add_argument("-s", "--sigma", type=float, help="sigma of noise to add to ptcloud", default=0.01)
    parser.add_argument("-N", "--numpts", type=int, help="number of points in synthetic ptcloud", default=150)
    parser.add_argument("-n", "--numreps", type=int, help="number of run repetitions", default=1)
    parser.add_argument("-r", "--runstats", help="when set, output stats on the run misfit", action="store_true", default=False)
    parser.add_argument("-i", "--infile", type=str, help="optional input filename of csv file with x,y,z ptcloud", default=None)
    parser.add_argument("-o", "--outfile", type=str, help="optional output filename of csv file of synthetically generated x,y,z ptcloud", default=None)
    parser.add_argument("-x", "--x", type=float, help="synth generated box center x coord", default=0.0)
    parser.add_argument("-y", "--y", type=float, help="synth generated box center y coord", default=10.0)
    parser.add_argument("-z", "--z", type=float, help="synth generated box center z coord", default=0.0)
    parser.add_argument("-Y", "--yaw", type=float, help="synth generated box yaw", default=20.0)
    parser.add_argument("-P", "--pitch", type=float, help="synth generated box pitch", default=0.0)
    parser.add_argument("-R", "--roll", type=float, help="synth generated box roll", default=0.0)
    parser.add_argument("-W", "--w", type=float, help="synth generated box width", default=2.0)
    parser.add_argument("-H", "--h", type=float, help="synth generated box height", default=2.0)  # note -h is --help so using cap H
    parser.add_argument("-L", "--l", type=float, help="synth generated box length", default=5.0)
    args = parser.parse_args()

    for i in range(args.numreps):
        demonstrate(x=args.x, y=args.y, z=args.z, w=args.w, h=args.h, l=args.l, \
                    yaw=args.yaw, pitch=args.pitch, roll=args.roll, infile=args.infile, \
                    outfile=args.outfile, sigma=args.sigma, N=args.numpts, runstats=args.runstats, \
                    plotbox=args.plotbox, plotpts=args.plotpts)
