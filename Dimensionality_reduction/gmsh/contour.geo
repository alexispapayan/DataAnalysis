SetFactory(" OpenCASCADE " );
// Coordinate points
Point(1) = { 1.2404765609582846,-0.05163554693823852,0,1 } ;
Point(2) = { 0.9979735019880477,0.5192104709992191,0,1 } ;
Point(3) = { 0.2134913886973711,0.4238687657345043,0,1 } ;
Point(4) = { -0.05208940759895149,1.1123815996951771,0,1 } ;
Point(5) = { -0.3634881342918874,0.396908812991862,0,1 } ;
Point(6) = { -0.7935806265918962,0.6497697275179641,0,1 } ;
Point(7) = { -1.0303314234728722,0.09177082561920603,0,1 } ;
Point(8) = { -0.6891629433273503,-0.4260349217907885,0,1 } ;
Point(9) = { -0.529203098529682,-0.9783127128288301,0,1 } ;
Point(10) = { 0.08059677991764659,-0.4975482107423261,0,1 } ;
Point(11) = { 0.5235599935852244,-1.173982694976771,0,1 } ;
Point(12) = { 0.37158080746893063,-0.13229361135031462,0,1 } ;
// Segments
Line(1)={ 1,2 } ;
Line(2)={ 2,3 } ;
Line(3)={ 3,4 } ;
Line(4)={ 4,5 } ;
Line(5)={ 5,6 } ;
Line(6)={ 6,7 } ;
Line(7)={ 7,8 } ;
Line(8)={ 8,9 } ;
Line(9)={ 9,10 } ;
Line(10)={ 10,11 } ;
Line(11)={ 11,12 } ;
Line(12)={ 12,1 } ;
// LineLoop
Line Loop(1)={1,2,3,4,5,6,7,8,9,10,11,12};
// Surface
Plane Surface(1)={ 1 } ;// Add transfinite lines
Transfinite Line {1}=1 Using Bump 1;
Transfinite Line {2}=1 Using Bump 1;
Transfinite Line {3}=1 Using Bump 1;
Transfinite Line {4}=1 Using Bump 1;
Transfinite Line {5}=1 Using Bump 1;
Transfinite Line {6}=1 Using Bump 1;
Transfinite Line {7}=1 Using Bump 1;
Transfinite Line {8}=1 Using Bump 1;
Transfinite Line {9}=1 Using Bump 1;
Transfinite Line {10}=1 Using Bump 1;
Transfinite Line {11}=1 Using Bump 1;
Transfinite Line {12}=1 Using Bump 1;
// Target edge length
Mesh.CharacteristicLengthFactor=0.2;