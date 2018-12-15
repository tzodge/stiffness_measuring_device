clc ;
clear;
% fig = figure;
% fig.Units = 'centimeters';
% fig.PaperType = 'usletter';
% pgon = polyshape([0 0 1 1].*20,[1 0 0 1].*20);
% 
% 
% set(gca,'units','centimeters')
% set(gca,'xlimmode','manual','ylimmode','manual')
% axpos = get(gca,'position');
% % axpos = [-1,-1,0,0];
% % set(gca,'position',[0 0 10  10])
% 
% plot(pgon);
% hold on
% plot([1,1],[1,2],'*r')
% 
% axis equal
% print('FillPageFigure','-dpdf');
% 
% 




x=7:0.05:15;
y=sin(x);
plot(x,y)
% Force MATLAB to render the figure so that the axes 
% are created and the properties are updated
drawnow  
% Define the axes' 'units' property
% Note: this does not mean that one cm of the axes equals 
%  one cm on the axis ticks.  The 'position' property 
%  will also need to be adjusted so that these match
set(gca,'units','centimeters')
% Force the x-axis limits and y-axis limits to honor your settings, 
% rather than letting the renderer choose them for you
set(gca,'xlimmode','manual','ylimmode','manual')
% Get the axes position in cm, [locationX locationY sizeX sizeY]
% so that we can reuse the locations
axpos = get(gca,'position');
% Use the existing axes location, and map the axes size (in cm) to the
%  axes limits so there is a true size correspondence
set(gca,'position',[axpos(1:2) abs(diff(xlim)) abs(diff(ylim))])
% Optional: Since we are forcing the x-axis limits and y-axis limits,
% the print out may not display the desired tick marks. In order to keep 
% these, you can select "File-->Preferences-->Figure Copy Template".  Then
% choose "Keep axes limits and tick spacing" in the "Uicontrols and axes"
% Frame.  Click on "Apply to Figure" and then "OK".
% Print the figure to paper in real size.
print
% Print to a file in real size and look at the result
print(gcf,'-dpng','-r0','sine.png')
% winopen('sine.png')