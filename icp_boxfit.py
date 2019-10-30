import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import numpy as np
import time
import icp


def Rmtx(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


def plot_ptcloud(ptcloud, color='k', ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ptcloud[:,0], ptcloud[:,1], ptcloud[:,2], c=color, marker='o')
    return ax


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


def generate_synth_data(x=2, y=0, z=0, w=2, h=2, d=2, yaw=10, pitch=45, roll=0, noise_sigma=0.01, N=150):
    # Presently doesn't handle scaling (w,h,d)
    # but there are other ICP algos that do which we can implement soon.

    # Generate an "input" ptcloud to which we'll want to fit a box.
    # convert from center position + extents to box bounds
    x1, x2, y1, y2, z1, z2 = xyzwhd_coords_to_x1y1(x,y,z, w,h,d)
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
    # np.random.shuffle(A)
    return A


def xyzwhd_coords_to_x1y1(x,y,z, w,h,d):
    x1 = x - w/2.0
    x2 = x + w/2.0
    y1 = y - h/2.0
    y2 = y + h/2.0
    z1 = z - d/2.0
    z2 = z + d/2.0
    return x1, x2, y1, y2, z1, z2


def demonstrate(x=5, y=0, z=0, w=2, h=2, d=4, yaw=10, pitch=0, roll=0, sigma=0.01, N=150, runstats=False, plotbox=False, plotpts=False):
    """Outermost wrapper demonstrating use of these functions estimating box fit."""

    # for now we don't yet have actual ptcloud input, so generate some
    A = generate_synth_data(x, y, z, w, h, d, yaw, pitch, roll, sigma, N)

    # Create a box-based ptcloud we'll rotate to fit the input ptcloud
    # B = np.copy(A)  # special case for testing - fit exact copy of data
    N = int(A.shape[0]/6)
    x0 = 0; y0=0; z0=0;  # initial estimates
    x1, x2, y1, y2, z1, z2 = xyzwhd_coords_to_x1y1(x0, y0, z0, w, h, d)
    B = generate_ptcloud_on_box(x1, x2, y1, y2, z1, z2, N)

    # Rotate/translate ptcloud B to fit ptcloud A
    # T is xform mtx of rotations (cols 0:2) & translations (col 3) to make B closest to A
    T, distances, iterations = icp.icp(B, A, tolerance=0.000001)
    angles = rotationMatrixToEulerAngles(T[:3, :3]) * 180/np.pi

    # Print table
    print('               %7s %7s %7s    %7s %7s %7s    %5s %5s %5s' %
        ('x', 'y', 'z',  'yaw', 'pitch', 'roll',  'w', 'h', 'd') )
    print('ground truth:  %7.2f %7.2f %7.2f    %7.2f %7.2f %7.2f    %5.2f %5.2f %5.2f' %
        (x, y, z, yaw, pitch, roll, w, h, d) )
    print('solution est:  %7.2f %7.2f %7.2f    %7.2f %7.2f %7.2f    %5.2f %5.2f %5.2f' %
        ( T[0,3], T[1,3], T[2,3],  angles[0], angles[1], angles[2],  w, h, d) )
    print(' ')

    if runstats:
        print('Run stats:')
        print('  num pts:', N)
        print('  sigma:', sigma)
        print(' ', iterations, 'iterations')
        print('  distances mean/std:', distances.mean(), distances.std())
        # # print('  T matrix:')
        # # print(T)

    if plotbox:
        # For testing create a new ptcloud C estimate of ptcloud A by xforming B, 
        # and plot on top of A to verify the box estimate in boxpts
        if plotpts:
            C = np.ones((N*6, 4))
            C[:,0:3] = np.copy(B)
            C = np.dot(T, C.T).T
            ax = plot_ptcloud(C, color='blue', ax=ax)

        # Plot original ptcloud
        ax = plot_ptcloud(A, color='red')
        # Plot the solution estimate box
        # (first xform the box by the ptcloud transform solution estimate)
        C = np.ones((8, 4))
        C[:,0:3] = np.copy(np.array([[x1,y1,z1], [x1,y1,z2], [x1,y2,z2], [x1,y2,z1],
                                     [x2,y1,z1], [x2,y1,z2], [x2,y2,z2], [x2,y2,z1]]))
        boxpts = np.dot(T, C.T).T
        plot_box(boxpts, ax=ax, col='black')
        plt.show()


def plot_box(boxpts, ax=None, col='black'):
    fullbox = np.concatenate((boxpts[:4,:],boxpts[0:1,:],boxpts[4:8,:],boxpts[4:5,:],
                              boxpts[5:6,:],boxpts[1:3,:],boxpts[6:8,:],boxpts[3:4,:]), axis=0)
    ax.plot(fullbox[:,0], fullbox[:,1], fullbox[:,2], c=col)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plotbox", help="plot soln box", action="store_true", default=False)
    parser.add_argument("-P", "--plotpts", help="plot soln box & ptcloud", action="store_true", default=False)
    parser.add_argument("-s", "--sigma", type=float, help="sigma of noise to add to ptcloud", default=0.01)
    parser.add_argument("-N", "--numpts", type=int, help="number of points in ptcloud", default=150)
    parser.add_argument("-r", "--runstats", help="number of points in ptcloud", action="store_true", default=False)
    args = parser.parse_args()

    demonstrate(x=0, y=10, z=0, w=3, h=4, d=5, yaw=10, pitch=0, roll=0, sigma=args.sigma, N=args.numpts, runstats=args.runstats, plotbox=args.plotbox, plotpts=args.plotpts)
