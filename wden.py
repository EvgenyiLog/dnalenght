import numpy as np

def wavedec(x, N, *args):
    """
    Multiple level 1-D discrete fast wavelet decomposition

    Calling Sequence
    ----------------
    [C,L]=wavedec(X,N,wname)
    [C,L]=wavedec(X,N,Lo_D,Hi_D)

    Parameters
    ----------
    wname : wavelet name, haar( "haar"), daubechies ("db1" to "db36"), coiflets ("coif1" to "coif17"), symlets ("sym2" to "sym20"), legendre ("leg1" to "leg9"), bathlets("bath4.0" to "bath4.15" and "bath6.0" to "bath6.15"), dmey ("dmey"), beyklin ("beylkin"), vaidyanathan ("vaidyanathan"), biorthogonal B-spline wavelets ("bior1.1" to "bior6.8"), "rbior1.1" to "rbior6.8"
    X : signal vector
    N : decompostion level
    Lo_D : lowpass analysis filter
    Hi_D : highpass analysis filter
    C : coefficient vector
    L : length vector

    Description
    -----------
    wavedec can be used for multiple-level 1-D discrete fast wavelet
    decompostion using a specific wavelet name wname or wavelet decompostion
    filters Lo_D and Hi_D. Such filters can be generated using wfilters.

    The global extension mode which can be change using dwtmode is used.

    The coefficient vector C contains the approximation coefficient at level N
    and all detail coefficient from level 1 to N

    The first entry of L is the length of the approximation coefficent,
    then the length of the detail coefficients are stored and the last
    value of L is the length of the signal vector.

    The approximation coefficient can be extracted with C(1:L(1)).
    The detail coefficients can be obtained with C(L(1):sum(L(1:2))),
    C(sum(L(1:2)):sum(L(1:3))),.... until C(sum(L(1:length(L)-2)):sum(L(1:length(L)-1)))

    Examples
    --------
    X = wnoise(4,10,0.5); //doppler with N=1024
    [C,L]=wavedec(X,3,'db2')
    """
    x = x.flatten()
    m1 = 1
    n1 = x.shape[0]
    if (len(args) == 1 and isinstance(args[0], str)):
        wname = args[0]
        ret = _wavelet_parser(wname.encode())
        filterLength = _wfilters_length(wname.encode())
        Lo_D, Hi_D = wfilters(wname,'d')
    elif(len(args) == 2):
        Lo_D = args[0]
        Hi_D = args[1]
        filterLength = Lo_D.shape[0]
    else:
        raise Exception("Wrong input!!")

    stride, val = _wave_len_validate(x.shape[0],filterLength)
    if (val == 0 or stride < N):
        raise Exception("Input signal is not valid for selected decompostion level and wavelets!")
    m4 = 1
    m5 = 1
    n4 = 0
    calLen = n1 * m1
    for count in np.arange(N):
        calLen += filterLength - 1
        temLen = np.floor(calLen/2).astype(int)
        n4 += temLen
        calLen = temLen

    n4 += temLen
    if (dwtmode("status","nodisp") == 'per'):
        n4 = 0
        calLen = n1 * m1
        for count in np.arange(N):
            # calLen += m3*n3 - 1;
            calLen = np.ceil(calLen/2.0).astype(int)
            temLen = calLen
            n4 += temLen
            # calLen = temLen;
        n4 += temLen
    n5 = N + 2

    output1 = np.zeros(n4*m4,dtype=np.float64)
    output2 = np.zeros(n5*m5,dtype=np.int32)
    _wave_dec_len_cal(filterLength, m1*n1, N, output2)
    _wavedec(x, output1, Lo_D, Hi_D, output2, N)
    return output1, output2


def waverec(C, L, *args):
    """
    Multiple level 1-D inverse discrete fast wavelet reconstruction

    Calling Sequence
    ----------------
    x0=waverec(C,L,wname)
    x0=waverec(C,L,Lo_R,Hi_R)

    Parameters
    ----------
    wname : wavelet name, haar( "haar"), daubechies ("db1" to "db36"), coiflets ("coif1" to "coif17"), symlets ("sym2" to "sym20"), legendre ("leg1" to "leg9"), bathlets("bath4.0" to "bath4.15" and "bath6.0" to "bath6.15"), dmey ("dmey"), beyklin ("beylkin"), vaidyanathan ("vaidyanathan"), biorthogonal B-spline wavelets ("bior1.1" to "bior6.8"), "rbior1.1" to "rbior6.8"
    x0 : reconstructed vector
    Lo_R : lowpass synthesis filter
    Hi_R : highpass synthesis filter
    C : coefficent array
    L : length array

    Description
    -----------
    waverec can be used for multiple-level 1-D inverse discrete fast wavelet
    reconstruction.

    waverec supports only orthogonal or biorthogonal wavelets.

    Examples
    --------
    X = wnoise(4,10,0.5); //doppler with N=1024
    [C,L]=wavedec(X,3,'db2');
    x0=waverec(C,L,'db2');
    err = sum(abs(X-x0))
    """
    C = C.flatten()
    m1 = 1
    n1 = C.shape[0]
    L = L.flatten()
    m2 = 1
    n2 = L.shape[0]
    L_summed_len = 0
    for count in np.arange(m2 * n2 - 1):
        L_summed_len += L[count]
    if (L_summed_len != m1*n1):
        raise Exception("Inputs are not coef and length array!!!")
    val = 0
    for count in np.arange(m2 * n2 - 1):
        if (L[count] > L[count+1]):
            val = 1
            break
    if (val != 0):
        raise Exception("Inputs are not coef and length array!!!")

    if (len(args) == 1 and isinstance(args[0], str)):
        wname = args[0]
        ret = _wavelet_parser(wname.encode())
        filterLength = _wfilters_length(wname.encode())
        Lo_R, Hi_R = wfilters(wname,'r')
    elif(len(args) == 2):
        Lo_R = args[0]
        Hi_R = args[1]
        filterLength = Lo_R.shape[0]
    else:
        raise Exception("Wrong input!!")
    if (L[0] < filterLength):
        raise Exception("Input signal is not valid for selected decompostion level and wavelets!\n")
    m4 = 1
    n4 = L[m2*n2-1]
    output1 = np.zeros(n4*m4,dtype=np.float64)
    _waverec(C, output1, Lo_R, Hi_R, L, m2*n2-2)
    return output1


def wrcoef(approx_or_detail, C, L, *args):
    """
    Restruction from single branch from multiple level decomposition

    Calling Sequence
    ----------------
    X=wrcoef(type,C,L,wname,[N])
    X=wrcoef(type,C,L,Lo_R,Hi_R,[N])

    Parameters
    ----------
    type : approximation or detail, 'a' or 'd'.
    wname : wavelet name
    X : vector of reconstructed coefficents
    Lo_R : lowpass synthesis filter
    Hi_R : highpass syntheis filter
    C : coefficent array
    L : length array
    N : restruction level with length(L)-2>=N

    Description
    -----------
    wrcoef is for reconstruction from single branch of multiple level
    decomposition from 1-D wavelet coefficients. Extension mode is stored as a global variable
    and could be changed with dwtmode. If N is omitted, maximum level (length(L)-2) is used.

    The wavelet coefficents C and L can be generated using wavedec.

    Examples
    --------
    x=rand(1,100)
    [C,L]=wavedec(x,3,'db2')
    x0=wrcoef('a',C,L,'db2',2)
    """
    raise Exception("Not yet implemented!!")


def appcoef(C, L, *args):
    """
    1-D approximation coefficients extraction

    Calling Sequence
    ----------------
    A=appcoef(C,L,wname,[N])
    A=appcoef(C,L,Lo_R,Hi_R,[N])

    Parameters
    ----------
    wname : wavelet name, haar( "haar"), daubechies ("db1" to "db20"), coiflets ("coif1" to "coif5"), symlets ("sym2" to "sym20"), legendre ("leg1" to "leg9"), bathlets("bath4.0" to "bath4.15" and "bath6.0" to "bath6.15"), dmey ("dmey"), beyklin ("beylkin"), vaidyanathan ("vaidyanathan"), biorthogonal B-spline wavelets ("bior1.1" to "bior6.8"), "rbior1.1" to "rbior6.8"
    A : extracted approximation coefficients
    Lo_R : lowpass synthesis filter
    Hi_R : highpass syntheis filter
    C : coefficent array
    L : length array
    N : restruction level with N<=length(L)-2

    Description
    -----------
    appcoef can be used for extraction or reconstruction of approximation
    coefficents at  level N after a multiple level decompostion.
    Extension mode is stored as a global variable and could be changed
    with dwtmode. If N is omitted, the maximum level (length(L)-2) is used.

    The length of A depends on the level N.
    C and L can be generated using wavedec.

    Examples
    --------
    X = wnoise(4,10,0.5)
    [C,L]=wavedec(X,3,'db2')
    A2=appcoef(C,L,'db2',2)
    """
    raise Exception("Not yet implemented!!")


def detcoef(C, L, N=None):
    """
    1-D detail coefficients extraction

    Calling Sequence
    ----------------
    D=detcoef(C,L,[N])

    Parameters
    ----------
    D : reconstructed detail coefficient
    C : coefficent array
    L : length array
    N : restruction level with N<=length(L)-2
    Description
    -----------
    detcoef is for extraction of detail coeffient at different level
    after a multiple level decompostion. Extension mode is stored as
    a global variable and could be changed with dwtmode. If N is omitted,
    the detail coefficients will extract at the  maximum level (length(L)-2).

    The length of D depends on the level N.

    C and L can be generated using wavedec.

    Examples
    --------
    X = wnoise(4,10,0.5); //doppler with N=1024
    [C,L]=wavedec(X,3,'db2');
    D2=detcoef(C,L,2)
    """
    C = C.flatten()
    m1 = 1
    n1 = C.shape[0]
    L = L.flatten()
    m2 = 1
    n2 = L.shape[0]

    L_summed_len = 0
    for count in np.arange(m2 * n2 - 1):
        L_summed_len += L[count]
    if (L_summed_len != m1*n1):
        raise Exception("Inputs are not coef and length array!!!")
    val = 0
    for count in np.arange(m2 * n2 - 1):
        if (L[count] > L[count+1]):
            val = 1
            break
    if (val != 0):
        raise Exception("Inputs are not coef and length array!!!")
    if (N is None):
        m4 = 1
        n4 = L[0]
        N = m2*n2 - 2
    else:
        if ((N > m2*n2 - 2) or N < 1):
            raise Exception("Level Parameter is not valid for input vector!!!")
        m4 = 1
        n4 = L[n2*m2 - N - 1]
    output1 = np.zeros(n4*m4,dtype=np.float64)
    _detcoef(C,L,output1,m2*n2-2,N)
    return output1

def wnoisest(C,L=None,S=None):
    """
    estimates of the detail coefficients' standard deviation for levels contained in the input vector S

    Parameters
    ----------
    C: array_like
         coefficent array
    L: array_like
         coefficent array
    S: array_like
         estimate noise for this decompostion levels
    Returns
    -------
    STDC: array_like
         STDC[k] is an estimate of the standard deviation of C[k]

    Examples
    --------
    [c,l] = wavedec(x,2,'db3')
    wnoisest(c,l,[0,1])
    """
    if (S is None and L is None):
        if (isinstance(C, list)):
            STDC = zeros(1,length(C))
            for k in np.arange(len(C)):
                STDC[k] = np.median(np.abs(C[k]))/0.6745
        else:
            STDC = np.median(np.abs(C))/0.6745
        return STDC

    maxLevel = np.size(L)-2
    if (S is None):
        S = np.arange(maxLevel)

    if(maxLevel < np.size(S)):
        print("C,L does not contain so much levels. reduce S!")
    STDC = np.zeros(np.size(S))
    for level in np.arange(np.size(S)):

        # STDC(level) = median(abs(C(sum(l(1:(maxLevel-level+1)))+1:sum(l(1:(maxLevel-level+2))))))/.6745;
        # STDC[level] = np.median(np.abs(C[maxLevel-S[level]]))/.6745
        STDC[level] = np.median(np.abs(detcoef(C,L,level + 1)))/.6745
        # STDC(level) = median(abs(C(sum(L(1:level))+(1:L(level+1)))))/.6745;

    return STDC


def wden(*args):
    """
    wden performs an automatic de-noising process of a one-dimensional signal using wavelets.
    Calling Sequence
    ----------------
    [XD,CXD,LXD] = wden(X,TPTR,SORH,SCAL,N,wname)
    [XD,CXD,LXD] = wden(C,L,TPTR,SORH,SCAL,N,wname)
    Parameters
    ----------
    x: array_like
          input vector
    C: array_like
         coefficent array
    L: array_like
         coefficent array
    TPTR: str threshold selection rule
         'rigrsure' uses the principle of Stein's Unbiased Risk.
         'heursure' is an heuristic variant of the first option.
         'sqtwolog' for universal threshold
         'minimaxi' for minimax thresholding
    SORH: str
         ('s' or 'h') soft or hard thresholding
    SCAL: str
         'one' for no rescaling
         'sln' for rescaling using a single estimation of level noise based on first-level coefficients
         'mln' for rescaling done using level-dependent estimation of level noise
    N: int
          N: decompostion level
    wname: str
          wavelet name
    Returns
    -------
    XD: array_like
         de-noised signal
    CXD: array_like
         de-noised coefficent array
    LXD: array_like
         de-noised length array

    Examples
    --------
    [xref,x] = wnoise(3,11,3)
    level = 4
    xd = wden(x,'heursure','s','one',level,'sym8')
    """
    if (len(args) == 6):
        x = args[0]
        TPTR = args[1]
        SORH = args[2]
        SCAL = args[3]
        N = args[4]
        wname = args[5]
        if (not isinstance(wname, str)):
            raise Exception("wname must be a string")
        C,L = wavedec(x,N,wname)
    elif (len(args) == 6):
        C = args[0]
        L = args[1]
        TPTR = args[2]
        SORH = args[3]
        SCAL = args[4]
        N = args[5]
        wname = args[6]
        if (not isinstance(wname, str)):
            raise Exception("wname must be a string")
    else:
        raise Exception("Wrong input!!")
    if (not isinstance(SCAL, str)):
        raise Exception("SCAL must be a string")
    if (not isinstance(SORH, str)):
        raise Exception("SORH must be a string")
    if (not isinstance(TPTR, str)):
        raise Exception("TPTR must be a string")

    if (SCAL == 'one'):
        sigma = np.ones(N)
    elif (SCAL == 'sln'):
        sigma = np.ones(N)*wnoisest(C,L,np.array([0]))
    elif (SCAL == 'mln'):
        sigma = wnoisest(C,L,np.arange(N))
    else:
        raise Exception("SCAL must be either ''one'',''sln'' or ''mln''")

    D = []
    for n in np.arange(N):
        D.append(detcoef(C,L,n + 1))
    CXD = C
    LXD = L
    i = 0
    for n in np.arange(N,0,-1)-1:
        i = i+1
        CXD[np.sum(L[:i])+np.arange(L[i])] = wthresh(D[n],SORH,thselect(D[n]/sigma[n],TPTR)*sigma[n])

    # for n in np.arange(N):
    #    CXD[N-n] = wthresh(C[N-n],SORH,thselect(C[N-n]/sigma[n],TPTR)*sigma[n])

    XD = waverec(CXD, LXD, wname)
    return XD,CXD,LXD


def thselect(X,TPTR):
    """
    Threshold selection for de-noising. The algorithm works only if the signal X has a white noise of N(0,1). Dealing with unscaled or nonwhite noise can be handled using rescaling of the threshold.

    Parameters
    ----------
    X: array
         input vector with scaled white noise (N(0,1))
    TPTR: str
         'rigrsure': adaptive threshold selection using principle of Stein's Unbiased Risk Estimate.
         'heursure': heuristic variant of the first option.
         'sqtwolog': threshold is sqrt(2*log(length(X))).
         'minimaxi': minimax thresholding.


    Returns
    -------
    THR: float
         threshold X-adapted value using selection rule defined by string TPTR

    Examples
    --------
    x = np.random.randn(1000)
    thr = thselect(x,'rigrsure')
    """
    if (TPTR == 'rigrsure'):
        THR = ValSUREThresh(X)
    elif (TPTR == 'heursure'):
        n,j = dyadlength(X)
        magic = np.sqrt(2*np.log(n))
        eta = (np.linalg.norm(X)**2 - n)/n
        crit = j**(1.5)/np.sqrt(n)
        if (eta < crit):
            THR = magic
        else:
            THR = np.min((ValSUREThresh(X), magic))
    elif (TPTR == 'sqtwolog'):
        n,j = dyadlength(X)
        THR = np.sqrt(2*np.log(n))
    elif (TPTR == 'minimaxi'):
        lamlist = [0, 0, 0, 0, 0, 1.27, 1.474, 1.669, 1.860, 2.048, 2.232, 2.414, 2.594, 2.773, 2.952, 3.131, 3.310, 3.49, 3.67, 3.85, 4.03, 4.21]
        n,j = dyadlength(X)
        if(j <= np.size(lamlist)):
            THR = lamlist[j-1]
        else:
            THR = 4.21 + (j-np.size(lamlist))*0.18
    return THR


def ValSUREThresh(X):
    """
    Adaptive Threshold Selection Using Principle of SURE

    Parameters
    ----------
    X: array
         Noisy Data with Std. Deviation = 1

    Returns
    -------
    tresh: float
         Value of Threshold

    """
    n = np.size(X)

    # a = mtlb_sort(abs(X)).^2
    a = np.sort(np.abs(X))**2

    c = np.linspace(n-1,0,n)
    s = np.cumsum(a)+c*a
    risk = (n - (2 * np.arange(n)) + s)/n
    # risk = (n-(2*(1:n))+(cumsum(a,'m')+c(:).*a))/n;
    ibest = np.argmin(risk)
    THR = np.sqrt(a[ibest])
    return THR


def dyadlength(x):
    """
    Find length and dyadic length of array

    Parameters
    ----------
    X: array
         array of length n = 2^J (hopefully)

    Returns
    -------
    n: int
         length(x)
    J: int
         least power of two greater than n
    """
    n = np.size(x)
    J = np.ceil(np.log(n)/np.log(2)).astype(np.int)
    return n,J


def wthresh(X,SORH,T):
    """
    doing either hard (if SORH = 'h') or soft (if SORH = 's') thresholding

    Parameters
    ----------
    X: array
         input data (vector or matrix)
    SORH: str
         's': soft thresholding
         'h' : hard thresholding
    T: float
          threshold value

    Returns
    -------
    Y: array_like
         output

    Examples
    --------
    y = np.linspace(-1,1,100)
    thr = 0.4
    ythard = wthresh(y,'h',thr)
    ytsoft = wthresh(y,'s',thr)
    """
    if ((SORH) != 'h' and (SORH) != 's'):
        print(' SORH must be either h or s')

    elif (SORH == 'h'):
        Y = X * (np.abs(X) > T)
        return Y
    elif (SORH == 's'):
        res = (np.abs(X) - T)
        res = (res + np.abs(res))/2.
        Y = np.sign(X)*res
        return Y


def wnoise(FUN,N,SQRT_SNR=1):
    """
    Noisy wavelet test data

    Parameters
    ----------
    FUN: str / int
        1 or 'blocks'
        2 or 'bumps'
        3 or 'heavy sine'
        4 or 'doppler'
        5 or 'quadchirp'
        6 or 'mishmash'
    N: int
         vector length of X = 2^N
    SQRT_SNR: float
          standard deviation of added noise

    Returns
    -------
    X: array_like
         test data
    XN: array_like
         noisy test data (rand(1,N,'normal') is added!)

    Examples
    --------
    [x,noisyx] = wnoise(4,10,7);
    """
    N = 2**N
    X = np.zeros(N)
    if (isinstance(FUN, str)):
        if (FUN == 'blocks'):
            FUN = 1
        elif(FUN == 'bumps'):
            FUN = 2
        elif(FUN == 'heavy sine'):
            FUN = 3
        elif(FUN == 'doppler'):
            FUN = 4
        elif(FUN == 'quadchirp'):
            FUN = 5
        elif(FUN == 'mishmash'):
            FUN = 6
    if (FUN == 1):
        tj = np.array([.1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
        hj = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
        for n in np.arange(N):
            t = (n-1)/(N-1)
            X[n] = np.sum(hj*((1+np.sign(t-tj))/2.))
    elif (FUN == 2):
        tj = np.array([.1, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
        hj = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
        wj = np.array([.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005])
        for n in np.arange(N):
            t = (n-1)/(N-1)
            X[n] = np.sum(hj*((1+np.abs((t-tj)/wj)**4.)**(-1)))
    elif (FUN == 3):
        for n in np.arange(N):
            t = (n-1)/(N-1)
            X[n] = 4*np.sin(4*np.pi*t)-np.sign(t-0.3)-np.sign(.72-t)
    elif (FUN == 4):
        eps = 0.05
        for n in np.arange(N):
            t = (n-1)/(N-1)
            X[n] = np.sqrt(t*(1-t))*np.sin(2*np.pi*(1-eps)/(t+eps))
    elif (FUN == 5):
        t = np.arange(N)/N
        X = np.sin((np.pi/3.) * t * (N * t**2.))
    elif (FUN == 6):
        t = np.arange(N)/N
        X = np.sin((np.pi/3) * t * (N * t**2))
        X = X + np.sin(np.pi * (N * .6902) * t)
        X = X + np.sin(np.pi * t * (N * .125 * t))
    if (SQRT_SNR != 1):
        X = X/np.std(X)*SQRT_SNR
    XN = X+np.random.randn(N)

    return X,XN