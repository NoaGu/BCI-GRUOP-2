n=1
for i=[4:4:120,126:4:242]
    
    train(n)=str2num(EEG_event(i).type)
    n=n+1
end
%%
left_ind=find(trainingVec==1)

left=MIData(left_ind,:,:);
evoked_left=squeeze(mean(left))';
figure()
plot(evoked_left)
%%
right_ind=find(trainingVec==2)

right=MIData(right_ind,:,:);
evoked_right=squeeze(mean(right))';
figure()
plot(evoked_right)
%%
no_ind=find(trainingVec==3)

no_move=MIData(no_ind,:,:);
evoked_no=squeeze(mean(no_move))';
figure()
plot(evoked_no)
%%
x=fft(left,626,3)
x=squeeze(mean(x))
figure()
xlabels=(1:626)*(125/626)'
plot(xlabels,(abs(x)))