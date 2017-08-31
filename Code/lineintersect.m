function [x,y]=lineintersect(l1,l2)
%This function finds where two lines (2D) intersect
%Each line must be defined in a vector by two points
%[P1X P1Y P2X P2Y], you must provide two has arguments
%Example:
%l1=[93 388 120 354];
%l2=[102 355 124 377];
%[x,y]=lineintersect(l1,l2);
%You can draw the lines to confirm if the solution is correct
%figure
%clf
%hold on
% line([l1(1) l1(3)],[l1(2) l1(4)])
% line([l2(1) l2(3)],[l2(2) l2(4)])
% plot(x,y,'ro') %this will mark the intersection point
%
%There is also included in another m file the Testlineintersect
%That m file can be used to input the two lines interactively
%
%Made by Paulo Silva
%22-02-2011

%default values for x and y, in case of error these are the outputs
x=nan;
y=nan;

%test if the user provided the correct number of arguments
if nargin~=2
disp('You need to provide two arguments')
return
end

%get the information about each arguments
l1info=whos('l1');
l2info=whos('l2');

%test if the arguments are numbers
if (~((strcmp(l1info.class,'double') & strcmp(l2info.class,'double'))))
disp('You need to provide two vectors')
return
end

%test if the arguments have the correct size
if (~all((size(l1)==[1 4]) & (size(l2)==[1 4])))
disp('You need to provide vectors with one line and four columns')
return
end

try
ml1=(l1(4)-l1(2))/(l1(3)-l1(1));
ml2=(l2(4)-l2(2))/(l2(3)-l2(1));
bl1=l1(2)-ml1*l1(1);
bl2=l2(2)-ml2*l2(1);
b=[bl1 bl2]';
a=[1 -ml1; 1 -ml2];
Pint=a\b;

%when the lines are paralel there's x or y will be Inf
if (any(Pint==Inf))
disp('No solution found, probably the lines are paralel')
return
end

%put the solution inside x and y
x=Pint(2);y=Pint(1);

%find maximum and minimum values for the final test
l1minX=min([l1(1) l1(3)]);
l2minX=min([l2(1) l2(3)]);
l1minY=min([l1(2) l1(4)]);
l2minY=min([l2(2) l2(4)]);

l1maxX=max([l1(1) l1(3)]);
l2maxX=max([l2(1) l2(3)]);
l1maxY=max([l1(2) l1(4)]);
l2maxY=max([l2(2) l2(4)]);

%Test if the intersection is a point from the two lines because 
%all the performed calculations where for infinite lines 
if ((x<l1minX) | (x>l1maxX) | (y<l1minY) | (y>l1maxY) |...
       (x<l2minX) | (x>l2maxX) | (y<l2minY) | (y>l2maxY) )
x=nan;
y=nan;
disp('There''s no intersection between the two lines')
return
end

catch err
%if some strange error ocurred show it to the user
rethrow(err)
disp('There''s no intersection between the lines (x=nan,y=nan)')
end