EEG = Data{1,1};
data = EEG(:,1:256);
nwin = 32;
nfft = nwin;
noverlap = 0.5 * nwin;
sample_freq = 256;


 [psd, f]=pwelch(data', nwin, noverlap, nfft, sample_freq);
 
 %%
rng default

n = 0:319;
x = cos(pi/4*n)+randn(size(n));

% Obtain the Welch PSD estimate using the default Hamming window and 
% DFT length. The default segment length is 71 samples and the DFT length is 
% the 256 points yielding a frequency resolution of  2*pi/sample. Because the
% signal is real-valued, the periodogram is one-sided and there are 256/2+1 
% points.
[pxx freq] = pwelch(x,256);
figure()
subplot(2,1,1)
plot(10*log10(pxx))
subplot(2,1,2)
plot(x)

