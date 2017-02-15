% script to demonstrate how to visualize the results from CRForest
% regression
close all
clear all
clc

GT = ReadMatFromFile('results\\GT.dat');
Pred = ReadMatFromFile('results\\Pred.dat');

minVal = min(GT, [], 2);
maxVal = max(GT, [], 2);

diag = minVal-2:0.5:maxVal+2

h = figure;
plot(GT, Pred, 'o');
hold on;
plot(diag, diag, 'k--', 'LineWidth', 2);
grid on;
xlabel('Ground Truth');
ylabel('Prediction');
title('Prediction Results');
print(h, 'visualizedResults.png', '-dpng');