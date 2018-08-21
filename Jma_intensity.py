def jma_int(x, y, z, dt):

    '''
    This function calculates the Japanese Meteorological Agency JMA, Intensity using 3 component
    acceleration data cm/s/s, gal.

    :param x: first component of waveform (gal)
    :param y: first component of waveform (gal)
    :param z: first component of waveform (gal)
    :param dt: interval between data points (seconds)
    :return: JMA seismic intensity

    Example
    x = np.linspace(1, 10, 1000)
    y = np.linspace(11, 20, 1000)
    z = np.linspace(21, 30, 1000)
    dt = 0.01

    this code is converted from Matlab by Dr. H. Serdar Kuyuk
    original code belongs to Shunpei, M., modified by Yu, Y.
    at International Research Institute of Disaster Science, Tohoku University, Japan
    serdarkuyuk@gmail.com
    https://scholar.google.co.jp/citations?user=ydxA_sAAAAAJ&hl=en
    '''

    import numpy as np


    data = np.matrix([x,y,z]).T
    nn=len(data)
    t0 =np.arange(0,nn*dt,dt)
    nt = 2
    while nn>nt:
        nt=nt*2

    data=np.vstack((data,np.zeros((nt-nn, 3))))
    fdata =np.fft.fft(data, axis=0)
    df =1/nt/dt
    nfold =nt/2+1

    f = np.arange(0, nfold*df, df)

    f0 = 0.5
    fc = 10.0
    f1 = np.zeros(int(nfold))
    f1[1::] = np.sqrt(1/f[1::])
    X = f/fc
    f2 = (1 + 0.694 * X**2 + 0.241 * X** 4 + 0.0577 * X**6 + 0.009664 * X**8
          + 0.00134 * X**10 + 0.000155 * X**12)**(-0.5)

    f3 = (1 - np.exp(-(f / f0)**3))**0.5
    fil = f1*f2*f3
    fildat1 = fdata[0:int(nfold), 0] * fil

    fildat2=fdata[0:int(nfold), 1] * fil
    fildat3=fdata[0:int(nfold), 2] * fil

    fildata=np.vstack((fildat1,fildat2,fildat3))
    data1=fildata.T
    data1[0, :] = complex(0, 0)

    data1=np.vstack((data1,np.flipud(data1[1:int(nfold)-1, :].conj())))
    t = np.arange(0,dt*nt,dt)
    data1 = np.fft.ifft(data1, axis=0)
    av = np.sqrt(np.real(data1[:,0])**2 + np.real(data1[:,1])**2 +np.real(data1[:,2])**2)
    avm = np.max(av)
    a1 = avm
    a2 = 0
    a3 = (a1 + a2) / 2
    lln = 0.3 / dt

    ll =0
    while ll != lln:
        ll = 0
        for i in range(nn):
            if av[i] >= a3:
                ll += 1

        if ll < lln:
            a1 = a3
            a3 = (a1 + a2) / 2
        elif ll > lln:
            a2 = a3
            a3 = (a1 + a2) / 2

        if ll == lln:
            break

    while ll == lln:
        ll = 0
        a4 = a3+0.1
        for k in range(nn):
            if av[k] >= a4:
                ll += 1

        if ll != lln:
            break
        a3 = a4

    sis=2*np.log10(a3)+0.94

    return sis
