function plotEstimate(myPrediction,score)
if myPrediction==1
prediction='prediction: left'

elseif myPrediction==2
        prediction='prediction: Right'

else
    prediction='None'
end
cla
    text(0.5,0.5 , prediction,...
        'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
    
     pause(1)
%      cla
%      text(0.5,0.5 , num2str(score),...
%         'HorizontalAlignment', 'Center', 'Color', 'white', 'FontSize', 40);
%      pause(1)
     cla
end