ts = linspace(-3,3, 16000);
omega = 30
w = @(ts) (exp(1i*4*pi*ts*omega) - exp(1i*2*pi*ts*omega)) ./ (1i*2*pi*ts*omega) ;
% w = @(ts) (exp(1i*4*pi*ts) - exp(1i*2*pi*ts)) ;
ws = w(ts);
figure(2)
% plot(ts,ws)
% plot(ts,imag(ws))
% plot(fft(ws))
x = ws;
% s = spectrogram(x,64,32,32,'yaxis');

% spectrogram(x,40,32,'yaxis')
% spectrogram(x,'yaxis')
spectrogram(x,512,256,256,16000,'yaxis')
% spectrogram(x,[],[],[],1600,'yaxis')
ax = gca;
ax.YScale = 'log';
figure(1)
plot(ts,ws)
%%
figure(3);
fnames = {
%     'sample/violin-cg-long.wav',
%     'sample/violin-g-low-long.wav',
%     'sample/violin-c-hi-long.wav',
    'sample/violin-spec.wav',
    'sample1/Track 1.wav',
    'sample1/Track 2.wav',
    'sample1/Track 3.wav',
    
};

i = 1
fname = fnames{1};
[xs,Fs] = audioread(fname);
Ts1 = xs(1*44100:4*44100,1);
x = Ts1;
spectrogram(x,64,60,128,44100,'yaxis')
% spectrogram(x,1512,400,256,44100,'yaxis')
% spectrogram(x,[],[],[],1600,'yaxis')
ax = gca;
ax.YScale = 'log';
