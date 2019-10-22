import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import numpy as np
import time
import icp


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


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


def generate_synth_data(x=0, y=0, z=0, w=2, h=2, d=2, yaw=0, pitch=0, roll=0):
    # Still working on this - translation and rotation will get replaced by
    # x,y,z and yaw,pitch,roll.  Present algo doesn't handle scaling (w,h,d)
    # but there are other ICP algos that do which we can implement soon.

    # Generate an "input" ptcloud to which we'll want to fit a box
    # convert from center position + extents to box bounds
    translation=0.5, rotation=0.7, noise_sigma=0.01, N=100
    A = generate_ptcloud_on_box(x1, x2, y1, y2, z1, z2, N)
    # Translate
    t = np.random.rand(3)*translation
    A += t
    # Rotate
    R = rotation_matrix(np.random.rand(3), np.random.rand() * rotation)
    A = np.dot(R, A.T).T
    # Add noise
    A += np.random.randn(N*6, 3) * noise_sigma
    # Shuffle to disrupt correspondence
    np.random.shuffle(A)
    return A


def xyzwhd_coords_to_x1y1(
    x1 = x - w/2.0
    x2 = x + w/2.0
    y1 = y - h/2.0
    y2 = y + h/2.0
    z1 = z - d/2.0
    z2 = z + d/2.0


def demonstrate():
    """Outermost wrapper demonstrating use of these functions estimating box fit."""

    A = generate_synth_data()

    # Create a box-based ptcloud we'll rotate to fit the input ptcloud
    #B = np.copy(A)
    x1, x2, y1, y2, z1, z2 = xyzwhd_coords_to_x1y1()
    B = generate_ptcloud_on_box(x1, x2, y1, y2, z1, z2, N)

    # Rotate/translate ptcloud B to fit ptcloud A
    # T is xform mtx of rotations (cols 0:2) & translations (col 3) to make B closest to A
    T, distances, iterations = icp.icp(B, A, tolerance=0.000001)

    # plot points
    ax = plot_ptcloud(A, color='red')
    #plot_box0(x1, x2, y1, y2, z1, z2, ax=ax, col='red')

    # Transform the box by the ptcloud xform solution
    C = np.ones((8, 4))
    C[:,0:3] = np.copy(np.array([[x1,y1,z1], [x1,y1,z2], [x1,y2,z2], [x1,y2,z1],
                                 [x2,y1,z1], [x2,y1,z2], [x2,y2,z2], [x2,y2,z1]]))
    boxpts = np.dot(T, C.T).T

    # Or for testing make new ptcloud C a homogeneous representation of ptcloud A by xforming B
    C = np.ones((N*6, 4))
    C[:,0:3] = np.copy(B)
    C = np.dot(T, C.T).T

    #ax = plot_ptcloud(C, color='blue', ax=ax)
    plot_box(boxpts, ax=ax, col='black')

    plt.show()


def plot_box(boxpts, ax=None, col='black'):
    fullbox = np.concatenate((boxpts[:4,:],boxpts[0:1,:],boxpts[4:8,:],boxpts[4:5,:],
                              boxpts[5:6,:],boxpts[1:3,:],boxpts[6:8,:],boxpts[3:4,:]), axis=0)
    ax.plot(fullbox[:,0], fullbox[:,1], fullbox[:,2], c=col)


if __name__ == "__main__":
    demonstrate()
