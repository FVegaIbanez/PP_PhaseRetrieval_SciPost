import numpy as np

def make_basis_h():
    # order = 12
    # ## cite the paper: Steepleton, Jacob. "Constructions of Hadamard Matrices." (2019).
    # # azimuthal
    # x = np.zeros(order-1)
    # for i in range(order-1):
    #     x[i] = i**2%(order-1)
    # x = x[x!=0]
    # x = np.unique(x)
    # A = np.eye(order)
    # A[:,0] = -1
    # A[0] = 1
    # Q = np.zeros((order-1,order-1))
    # Q[0,0] = 1
    # for i in range(1,order-1):
    #     if i in x:
    #         Q[0,i] = -1
    #     else:
    #         Q[0,i] = 1
    # for i in range(1,order-1):
    #     Q[i] = np.append(Q[i-1][-1:], Q[i-1][:-1])
    # A[1:,1:] = Q


    A = np.array([
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [-1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [-1., -1., -1.,  1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1.],
        [-1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1.],
        [-1., -1.,  1., -1.,  1.,  1.,  1., -1., -1.,  1., -1.,  1.],
        [-1.,  1., -1.,  1.,  1., -1.,  1., -1., -1., -1.,  1.,  1.],
        [-1.,  1.,  1.,  1., -1., -1.,  1., -1.,  1.,  1., -1., -1.],
        [-1., -1.,  1.,  1.,  1., -1., -1.,  1., -1.,  1.,  1., -1.],
        [-1.,  1., -1., -1.,  1.,  1., -1., -1.,  1.,  1.,  1., -1.],
        [-1.,  1.,  1., -1.,  1., -1., -1.,  1.,  1., -1., -1.,  1.],
        [-1.,  1.,  1., -1., -1.,  1.,  1.,  1., -1., -1.,  1., -1.],
        [-1., -1.,  1.,  1., -1.,  1., -1., -1.,  1., -1.,  1.,  1.],
        ])


    # radial
    R = np.array([
        [1,1,1,1],
        [1,1,-1,-1],
        [1,-1,1,-1],
        [1,-1,-1,1]
    ])
    # basis
    basis = np.zeros((48,48))
    cnt = 0
    for r in R:
        for a in A:
            basis[cnt] = np.array([rr * a for rr in r]).flatten()
            cnt += 1
    basis[basis==1] = 0
    basis[basis==-1] = np.pi
    return twopi_2_0(basis)
def make_basis_f():
    # azimuthal
    A = np.zeros(( 12, 12 ))
    for n in range(1,12):
        A[n,:] =np.linspace(0, 1, 12, endpoint=False)*(n) +  A[0,:]
    A = np.exp(1j*2*np.pi*A) # regulate range between -1 and 1
    # radial
    R = np.zeros(( 4, 4 ))
    for n in range(1,4):
        R[n,:] = np.linspace(0, 1, 4, endpoint=False)*(n) +  R[0,:]
    R = np.exp(1j*2*np.pi*R) # regulate range between -1 and 1
    # basis
    basis = np.zeros((48,48), dtype='complex')
    cnt = 0
    for r in R:
        for a in A:
            basis[cnt] = np.array([rr * a for rr in r]).flatten()
            cnt += 1
    return twopi_2_0(np.angle(basis))

def twopi_2_0(basis):
    basis[basis==np.pi*2] = 0
    return basis