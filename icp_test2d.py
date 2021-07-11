import numpy as np
import icp


def test2d(eps=1e-7, angle=0, offset=[0, 0], tol=1e-7):
    """
    Test out the icp() implementation on a simple 2D problem, by taking an
    example trio of points, rotating and translating them and adding noise,
    and then using icp() to estimate that rotation and translation.
    
    Parameters:
    -----------
    eps:  float:  stdev of gaussian noise added to the transformed points
    angle:  float:  angle in degrees of rotation
    offset:  list of floats:  x & y offsets
    tol:  float:  tolerance value passed to the icp() call
    """

    # arbitrarily chosen example trio of input points.
    # we'll rotate and translate these and try to solve for that rot & trans.
    A = np.asarray([[10, 0], [0, 10], [10, 10]], dtype=np.float64)
    print(f'A({A.shape}):\n',A)
    print('')

    # generate the 2d rotation matrix based on the input angle
    R = np.asarray([[np.cos(angle*np.pi/180), -np.sin(angle*np.pi/180)], [np.sin(angle*np.pi/180), np.cos(angle*np.pi/180)]])
    print(f'R({R.shape}):\n',R)
    print('')

    # rotate & translate the input points A, and add noise
    B = (R @ A.T).T + offset + eps*np.random.randn(*A.shape)
    print(f'B({B.shape}):\n',B)
    print('')

    # use the icp() call to estimate the rotation and translation
    T, distances, iterations, bulk_error = icp.icp(A, B, tolerance=tol, max_iterations=50)

    print(f'T({T.shape}):\n',T)
    print('')
    print('angle (deg) + offset:\n')
    print(np.arctan2(T[1,0], T[0,0])*180/np.pi, T[0,2], T[1,2])
    print('')
    print('distances, iterations, bulk_error:\n')
    print(distances, iterations, bulk_error)
