SetFactory(" OpenCASCADE " );
// Coordinate points
Point(1) = { 1.1055750951735361,0.08505172798861153,0,1 } ;
Point(2) = { 1.0595269823024784,0.46563413535565384,0,1 } ;
Point(3) = { 0.5759087535286743,0.808579571623018,0,1 } ;
Point(4) = { -0.2166909096432264,0.8492046234825806,0,1 } ;
Point(5) = { -0.6327747810721499,0.7175037763783794,0,1 } ;
Point(6) = { -0.9664903088945263,0.6571628791071722,0,1 } ;
Point(7) = { -0.7763472802602845,0.02582085132334226,0,1 } ;
Point(8) = { -0.6607162204886317,-0.305209426676841,0,1 } ;
Point(9) = { -0.5783935695912633,-0.9881216261919065,0,1 } ;
Point(10) = { 0.05273811669177748,-0.9783529633050528,0,1 } ;
Point(11) = { 0.48802859166593116,-0.8810140296717536,0,1 } ;
Point(12) = { 0.5496355305876836,-0.45625951941320325,0,1 } ;
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
Mesh.CharacteristicLengthFactor=0.9;