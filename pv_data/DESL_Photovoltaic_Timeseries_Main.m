%% loads and plot the GHI and PV time-series
% DESL-EPFL
% Rahul Gupta


clear all
clc

unzip('DESL_GHI_PV-Power_2017-11-24_2023-01-22_5minutes.zip','myfolder')

PVData = importdata('myfolder/DESL_GHI_PV-Power_2017-11-24_2023-01-22_5minutes.csv');

figure
subplot(2,1,1)
ghi = PVData.data(:,1);
ghi(isnan(ghi))=0;
csvwrite('ghi.csv', ghi)

plot(ghi)
ylabel('GHI (Watts/m2')
xlabel('time index')

subplot(2,1,2)
pv_gen = PVData.data(:,2);
pv_gen(isnan(pv_gen))=0;
csvwrite('pv_gen.csv', pv_gen)
type('pv_gen.csv')

plot(pv_gen)
ylabel('PV generation (Watts')
xlabel('time index')
