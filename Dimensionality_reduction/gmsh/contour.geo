SetFactory(" OpenCASCADE " );
// Coordinate points
Point(1) = { 1.0408641087455663,-0.10302270119033043,0,1 } ;
Point(2) = { 0.4204805905855329,0.45060826843860313,0,1 } ;
Point(3) = { -0.07548484725480818,0.9862254045849879,0,1 } ;
Point(4) = { -0.45504499451552033,0.6201203129952261,0,1 } ;
Point(5) = { -1.1050574660454455,0.2925036957176972,0,1 } ;
Point(6) = { -0.6229061402175986,-1.061766697638888,0,1 } ;
Point(7) = { -0.059841037747307174,-0.5610006868742673,0,1 } ;
Point(8) = { 0.8569897864495798,-0.6236675960330283,0,1 } ;
// Segments
Line(1)={ 1,2 } ;
Line(2)={ 2,3 } ;
Line(3)={ 3,4 } ;
Line(4)={ 4,5 } ;
Line(5)={ 5,6 } ;
Line(6)={ 6,7 } ;
Line(7)={ 7,8 } ;
Line(8)={ 8,1 } ;
// LineLoop
Line Loop(1)={1,2,3,4,5,6,7,8};
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
// Target edge length
Mesh.CharacteristicLengthFactor=0.8;