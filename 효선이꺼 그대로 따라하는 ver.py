import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.fft as fft
import astropy.constants as const
import astropy.units as u #####이건 내가 효선이 코드 218행의 H0의 단위 제대로 하기 위해 추가한 것
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from astropy.cosmology import LambdaCDM
from photutils.segmentation import detect_threshold
from photutils.segmentation import detect_sources
from photutils.segmentation import deblend_sources
from photutils.segmentation import SourceCatalog
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit
from math import pi

#----- Load Data
fitsdata = 'distorted_field.fits'
image = fits.open(fitsdata)
data = image[0].data


#----- Detect Source
threshold = detect_threshold(data, nsigma=2.)
sigma = 1.2
kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
kernel.normalize()
segm = detect_sources(data, threshold, npixels=5, kernel=kernel)
segm_deblend = deblend_sources(data, segm, npixels=5, kernel= kernel, 
                                nlevels=32, contrast=0.001)
sources = segm_deblend.data


#----- Detected Source Plot
plt.figure(figsize=(10,10))
plt.imshow(sources)


#----- Morphological Properties
cat = SourceCatalog(data, segm_deblend)
tbl = cat.to_table()

xcen = np.array(tbl['xcentroid'].data)
ycen = np.array(tbl['ycentroid'].data)
a = np.array(tbl['semimajor_sigma'].data)
b = np.array(tbl['semiminor_sigma'].data)
phi = np.array(tbl['orientation'].data) * pi / 180

e = (a-b)/(a+b)
e1 = e * np.cos(phi) #####사실 이건 e_x라고 해야한다. 수업 시간에 배운 e2랑은 다른 것이기 때문. 오로지 whisker plot을 그릴 때 20by20으로 만들어주기 위함이다. e의 정보를 다시 되살려야하므로.
e2 = e * np.sin(phi) #####사실 이건 e_y라고 해야한다. 수업 시간에 배운 e2랑은 다른 것이기 때문. 오로지 whisker plot을 그릴 때 20by20으로 만들어주기 위함이다. e의 정보를 다시 되살려야하므로.

real_e1 = e * np.cos(2*phi) #####이게 진짜 e1
real_e2 = e * np.sin(2*phi) #####이게 진짜 e2

#----- Whisker plot
background = np.zeros((5000, 5000))
start_loc = np.empty((0,2))           #np.empty((0,2))의 결과는 그냥 []이다. 어차피 추가할 목적으로 빈 어레이를 할당하는 것이기에 이렇게 첫 번째 원소에 0을 넣는 것이다. 우리는 아래로 원소를 추가해야 하니까 np.append를 쓸 때 2차원 행방향을 의미하는 axis=0을 써야한다.
end_loc = np.empty((0,2))             #np.zeros보다 np.empty를 쓰는 이유는 empty가 훨씬 빠르기 때문인 것 같다.
real_end_loc = np.empty((0,2))
e1_list = np.array([])
e2_list = np.array([])
#####phi_from_ex = np.array([])
#####phi_from_e1 = np.array([])
real_e1_list = np.array([])
real_e2_list = np.array([])
#####e_list = np.array([])

for i in range(20):
    for j in range(20):
        ind = np.where( (xcen>=i*250) & (xcen<(i+1)*250) & (ycen>=j*250) & (ycen<(j+1)*250) )
        e1_avg = 5*1e2*np.mean(e1[ind]) #? 왜 500을 곱하지? --> 아마 그냥 시각적인 크기를 키우려고?
        e2_avg = 5*1e2*np.mean(e2[ind])
        real_e1_avg = 5*1e2*np.mean(real_e1[ind])
        real_e2_avg = 5*1e2*np.mean(real_e2[ind])
        e1_list = np.append(e1_list, e1_avg)
        e2_list = np.append(e2_list, e2_avg)
        real_e1_list = np.append(real_e2_list, real_e1_avg)
        real_e2_list = np.append(real_e2_list, real_e2_avg)
        #####e_list = np.append(e_list, (e1_avg**2 + e2_avg**2)**(1/2))
        #####phi_from_ex =  np.append(phi_from_ex, np.arccos(e1_avg/e_list))
        #####phi_from_e1 =  np.append(phi_from_e1, (1/2)*np.arccos(real_e1_avg/e_list))
        center = np.array([[i*250+125, j*250+125]]) # 왜 []이걸 한 번이 아니라 두 번 감싸지? --> np.append 쓸 때 원래의 차원인 2차원 유지시켜주려면 추가하려는 array 또한 같은 차원인 2차원이어야 한다. 그래야지만 axis라는 파라미터 쓸 수 있다.
        start_loc = np.append(start_loc, center, axis=0)
        stick_end = np.array([[center[0,0]+e1_avg, center[0,1]+e2_avg]])
        #####stick_end = np.array([[center[0,0]+ e_list * np.cos(phi_from_ex) , center[0,1]+ e_list * np.sin(phi_from_ex)]])
        end_loc = np.append(end_loc, stick_end, axis=0)

fig = plt.figure(figsize=(10,10))
plt.imshow(background, cmap='gray', origin='lower') ##### 아, 이건 단순히 검은 바탕 만드려고 쓴거다. 아래 plot의 color='w'(즉, 하얀색)로 하여 검은 바탕에 흰 수염으로 그려지도록 함.
                                                    ##### imshow 빼려면 그냥 color = 'black'으로 하면 된다. 참고로 color 지정 안하면 색이 한 줄마다 다 다르게 나와서 정신없음.
plt.plot([start_loc[:,0], end_loc[:,0]], [start_loc[:,1], end_loc[:,1]], linewidth=0.5, color='w')
plt.xlabel("X (pixels)", size=15)
plt.ylabel("Y (pixels)", size=15)
plt.show()


#----- Mass Redconstruction # 아.. 원래의 이미지에서 100by100으로 만드는게 아니라 20by20으로 줄인 whisker plot에서 다시 100by100으로 키우는 것이다. interpolaiton을 이용하여.
##### 수정: 굳이 interpolation하지 않고 원 데이터 그대로 가져다 쓴다.
##### 효선이는 e1, e2를 써야하는 부분에서 e_x, e_y를 써서 틀렸다.
#f_e1 = interp2d(start_loc[:,0], start_loc[:,1], e1_list, kind='cubic') # 여기서 원래는 ramma1, ramma2 써야하는데 근사적으로 ramma1 = e1, ramma2 = e2라고 보고 있으므로,
#f_e2 = interp2d(start_loc[:,0], start_loc[:,1], e2_list, kind='cubic') # 그냥 e1, e2라고 적은 것이다.
f_e1 = interp2d(start_loc[:,0], start_loc[:,1], real_e1_list, kind='cubic') # 여기서 원래는 ramma1, ramma2 써야하는데 근사적으로 ramma1 = e1, ramma2 = e2라고 보고 있으므로,
f_e2 = interp2d(start_loc[:,0], start_loc[:,1], real_e2_list, kind='cubic') # 그냥 e1, e2라고 적은 것이다.

x = np.linspace(0,4999,250)# 이제 우리의 목적은 이 ramma1과 ramma2를 FT 시킨 것과, 그로부터나온 k1, k2로부터
y = np.linspace(0,4999,250)# kappa를 구하는 것이다.

e1_interp = f_e1(x,y)
e2_interp = f_e2(x,y)

Fr1 = fft.fft2(e1_interp)
Fr2 = fft.fft2(e2_interp)

k1 = fft.fftfreq(Fr1.shape[0], d=1) # ramma1이든 ramma2이든 k1과 k2는 모두 deflection potential에서 나온 것이다(1116 강의 [83] 참고).
k2 = fft.fftfreq(Fr2.shape[1], d=1) # 따라서 둘다 Fr1(즉, Fr1.shape[0]과 Fr1.shape[1])으로 해도 상관없다.

K1, K2 = np.meshgrid(k1, k2)
K = np.sqrt(K1**2 + K2**2) +1e-7 # 분모가 0으로 들어가므로 에러 방지하기 위한 듯 하다.

Fkappa = K**(-2) * ((K1**2-K2**2)*Fr1 + 2*K1*K2*Fr2)

kappa = fft.ifft2(Fkappa)
kappa = np.abs(kappa)

fig = plt.figure(figsize=(10,10))
plt.imshow(kappa, cmap='jet', origin='lower')
plt.show()

##### 성공!!!!!

#----- Tangential Shear
origin = [2500, 2500]
dx = xcen - origin[0]
dy = ycen - origin[1]

# Ellipticity Rotation
radius = np.sqrt(dx**2 + dy**2)
cos_phi = dx/radius
sin_phi = dy/radius
cos_2phi = cos_phi**2 - sin_phi**2
sin_2phi = 2*cos_phi*sin_phi
r_t = -real_e1*cos_2phi - real_e2*sin_2phi ###### 효선이는 여기서 잘못했다.
r_x = -real_e1*sin_2phi + real_e2*cos_2phi ###### phi가 아니라 2phi로부터 정의된 real_e1, real_e2(53,54행 참고)를 써야하는게 자명하다. e1과 e2의 정의에 의해서.

# Binning
gamma_t = []
gamma_terr = []
gamma_x = []
gamma_xerr = []

for i in range(14):
    ind = np.where((radius>=i*178) & (radius<(i+1)*178))
    r_t_avg = np.mean(r_t[ind])
    r_t_std = np.std(r_t[ind]) / np.sqrt(len(r_t[ind]))
    r_x_avg = np.mean(r_x[ind])
    r_x_std = np.std(r_x[ind]) / np.sqrt(len(r_x[ind]))#####효선이 실수했다. r_x로 해야되는데 r_t로 했다.
    gamma_t.append(r_t_avg)
    gamma_terr.append(r_t_std)
    gamma_x.append(r_x_avg)
    gamma_xerr.append(r_x_std)
print(gamma_x)    
print(gamma_xerr)    

R = np.arange(89, 2500, 178)
fig = plt.figure(figsize=(10,10))
plt.errorbar(R, gamma_x, yerr=gamma_xerr, c='y', marker='D', mfc='w', capsize=3, label=r'$\gamma_X$')
plt.errorbar(R, gamma_t, yerr=gamma_terr, c='k', marker='o', capsize=3, label =r'$\gamma_T$')
plt.axhline(0, linestyle=':')
plt.legend(fontsize=15)
plt.xlabel("Radius [pixel]", size=15)
plt.ylabel("Shear", size=15)
plt.show()


#----- SIS fitting
def SIS(x, r_E): 
    result = copy.deepcopy(x)
    inner = np.where(x<=r_E)
    outer = np.where(x>r_E)
    result[inner] = (2*x[inner]-r_E)/r_E #####pdf에 나온 것은 [r_E]단위이므로 분자분모에 [pixel]/[r_E]를 곱해준 것이다.
    result[outer] = r_E/(2*x[outer]-r_E)
    return result

pix_scale = 0.05    #[arcsec/pixel]

popt, pcov = curve_fit(SIS, R, gamma_t, sigma=gamma_terr, p0=(100))
#####r_E = popt[1] * pix_scale #####효선이가 틀린 것 같다. r_E는 [1]을 써야지 [0]은 x잖아..
#####r_Eerr = np.sqrt(pcov[1][1]) * pix_scale #####같은 이유로 이것도 [0]이 아니라 [1][1]일텐데.. [0]은 [x의 분산, x와 E_r의 공분산]이잖아..
#####게다가 error는 pcov에 sqrt를 취해야 하는데 효선이는 그냥 분산을 적었다. 이건 공식문서의 pcov 설명란에 "standard deviation error"를 구하는 방법이라고 정확히 명시되어있다.
#####위의 내 추측이 맞다는 정확적 증거로, 효선이는 그 다음에 r_Eerr을 활용하지 않았다. 오직 print로만 활용함.
##### !!!!! 내가 틀렸다. 이건 이상하게 popt가 값이 2개가 아니라 1개가 나와서 matrix가 2x2가 아니라 1x1이다... 그래서 popt랑 pcov 둘 다 효선이가 맞다. 
##### 단 np.sqrt()해줘야하는 건 내가 맞다.
##### !!!!! 아~ 이해했다.. def에서 첫 번째 들어가는 값은 xdata이고, curve_fit으로 추정하는 값은 그 다음에 위치하는 *params이다. 
r_E = popt[0] * pix_scale
r_Eerr = np.sqrt(pcov[0][0]) * pix_scale
##### 이건 효선이가 틀린게 맞다. r_Eerr은 내가 바로 윗줄에 적은 것 처럼, np.sqrt(pcov[0])이다.
##### 또 pcov는 2d array이므로 1x1이라도 인덱스를 두 개([0][0]) 써주어야 한다.

print("r_E = " + str(r_E) + "[arcsec]")
print("r_E_error = " + str(r_Eerr) + "[arcsec]")

#----- NFW fitting
def f(r, r_s):
    x = r/r_s #####이렇게 해줌으로써 위의 SIS 때와는 다르게 함수 집어넣을 때 SIS(x,r_E/pix_scale)이렇게 할 필요 없이 NFW(x,popt[0],popt[1])<<효선이코드 255행>> 이렇게 바로 Einstein radius를 집어넣어줄 수 있게 된다.
    result = copy.deepcopy(x)
    inner = np.where(x<1)
    critical = np.where(x==1)
    outer = np.where(x>1)
    x_in = x[inner]
    x_out = x[outer]
    result[inner] = 1/(x_in**2-1) * (1-2*np.arctanh(np.sqrt((1-x_in)/(1+x_in)))/np.sqrt(1-x_in**2))
    result[critical] = 1/3
    result[outer] = 1/(x_out**2-1) * (1-2*np.arctan(np.sqrt((x_out-1)/(1+x_out)))/np.sqrt(x_out**2-1))
    return result

def j(r, r_s):
    x = r/r_s
    result = copy.deepcopy(x)
    inner = np.where(x<1)
    critical = np.where(x==1)
    outer = np.where(x>1)
    x_in = x[inner]
    x_out = x[outer]
    result[inner] = 4*np.arctanh(np.sqrt((1-x_in)/(1+x_in)))/(x_in**2*np.sqrt(1-x_in**2)) + 2*np.log(x_in/2)/x_in**2 - 1/(x_in**2-1) + 2*np.arctanh(np.sqrt((1-x_in)/(1+x_in)))/((x_in**2-1)*np.sqrt(1-x_in**2))
    result[critical] = 2*np.log(1/2)+5/3
    result[outer] = 4*np.arctan(np.sqrt((x_out-1)/(1+x_out)))/(x_out**2*np.sqrt(x_out**2-1)) + 2*np.log(x_out/2)/x_out**2 - 1/(x_out**2-1) + 2*np.arctan(np.sqrt((x_out-1)/(1+x_out)))/(x_out**2-1)**(3/2)
    #####내꺼 어딘가 타이포가 난 것 같다.
    #####[inner]와 [outer] 각각 첫째항의 분모에, 2뒤에 np.sqrt 안 붙여줘서 결과 잘못나온 것.
    return result

def NFW(r, r_s, c):
    delta_c = (200/3) * (c**3/(np.log(1+c)-c/(1+c)))
    Kk = (2*r_s*delta_c*rho_c)/smass_c
    K = Kk*f(r, r_s)
    Gamma = Kk*j(r, r_s)
    g = Gamma/(1-K) #####이거 효선이는 이것의 역수로 해서 잘못된 결과가 나왔다.
    ind = np.where(abs(g)>1)
    g[ind] = 1/g[ind]
    return g

Omegam, OmegaL = 0.3, 0.7
zl = 0.5
zs = 1.0

G = const.G.value ##### 위의 gamma도 똑같은 G로 썼는데, flow 따라가보니 상관없긴해도 중복되니 위의 것을 Gamma로 바꿨다.
v_c = const.c.value
Mpc_to_m = const.pc.value * 1e6
m_to_pc = const.pc.value ** -1
arcsec_to_rad = 1/3600 * np.pi/180
H0 = ( 73 * ( (u.km/u.s) / (1e6*u.pc) ) ).decompose().value #####나는 astropy.units을 활용했다.
H = np.sqrt(H0**2 * (Omegam*(1+zl)**3 + OmegaL)) #####이건 lens의 z를 써야한다.

cosmo = LambdaCDM(H0=73, Om0=Omegam, Ode0=OmegaL)

Dd = cosmo.angular_diameter_distance(zl).value * Mpc_to_m
Ds = cosmo.angular_diameter_distance(zs).value * Mpc_to_m
Xl = cosmo.comoving_distance(zl).value * Mpc_to_m
Xs = cosmo.comoving_distance(zs).value * Mpc_to_m
Dds = (Xs-Xl) / (1+zs) ##### Angular diameter distance로 구한 lens와 source 사이의 거리이다.
D = Dd*Ds/Dds ##### 이건 Time delay 파트에서 쓰이는고, 여기서는 안쓰여서 정의해 줄 필요 없었다.

rho_c = 3*H**2/ (8*pi*G)
smass_c = v_c**2 / (4*pi*G) * Ds/(Dd*Dds)

popt, pcov = curve_fit(NFW, R, gamma_t, sigma=gamma_terr, p0=(411,2.5e7))

r_s = popt[0] * pix_scale * arcsec_to_rad * Dd ##### Angular Daimeter Distance의 정의에 의해.
r_serr = np.sqrt(pcov[0][0]) * pix_scale * arcsec_to_rad * Dd ##### 위에서 196행과 같이 np.sqrt를 해주어야한다.
c = popt[1]
c_err = np.sqrt(pcov[1][1]) ##### 위와 마찬가지로 분산이므로 sqrt 취해야한다.
r_200 = c*r_s
r_200_err = r_s*c_err + c*r_serr ##### ??? error가 이런식으로 전파되나?? ---> 궁금한 지점에 적은 것 참고. 맞는 것 같다.

M_200 = 4*pi/3 * r_200**3 * 200 * rho_c / (2*10**30)
M_200_err = 4*pi/3 * r_200_err**3 * 200 * rho_c / (2*10**30)

print("c = " + str(round(c,0)))
print("c_err = " + str(round(c_err,0)))
print("M_200 = " + str(round(M_200,-30)) + "[M\u2299]")
print("M_200_err = " + str(round(M_200_err, -30)) + "[M\u2299]")
print("r_s = " + str(round(r_s * m_to_pc / 1e6, 3)) + "[Mpc]")
print("r_serr = " + str(round(r_serr * m_to_pc / 1e6,3)) + "[Mpc]")

x = np.linspace(1,2500,1000)
fig = plt.figure(figsize=(10,10))
plt.plot(x,SIS(x,r_E/pix_scale), label="SIS")
plt.plot(x,NFW(x,popt[0],popt[1]), label="NFW")
plt.errorbar(R, gamma_t, yerr=gamma_terr, c='k', marker='o', capsize=3, label=r'$\gamma_T$')
plt.errorbar(R, gamma_x, yerr=gamma_xerr, c='y', marker='D', mfc='w', capsize=3, label=r'$\gamma_X$') ##### 1123 pdf 끝부분보면 교수님은 gamma_X의 error값도 넣주셨는데 효선이는 넣지 않아서 내가 임의로 넣었다.
plt.ylim(-0.2,0.4)#plt.ylim(-0.15,0.15) #####범위 교수님과 같게 하였다.
plt.axhline(0, linestyle=':')
plt.legend(fontsize=15)
plt.xlabel("Radius [pixel]", size=15)
plt.ylabel("Shear", size=15)
plt.show()

# 드디어 끝. 2022_0205_02:32(AM).
# 인줄 알았는데 228행에서 언급한 것처럼 타이포나서 고침
# 진짜 끝. 2022_0205_03:08(AM).