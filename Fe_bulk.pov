#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White}
camera {orthographic
  right -11.06*x up 11.80*y
  direction 1.00*z
  location <0,0,50.00> look_at <0,0,0>}
light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}

#declare simple = finish {phong 0.7}
#declare pale = finish {ambient .5 diffuse .85 roughness .001 specular 0.200 }
#declare intermediate = finish {ambient 0.3 diffuse 0.6 specular 0.10 roughness 0.04 }
#declare vmd = finish {ambient .0 diffuse .65 phong 0.1 phong_size 40. specular 0.500 }
#declare jmol = finish {ambient .2 diffuse .6 specular 1 roughness .001 metallic}
#declare ase2 = finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.70 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient .15 brilliance 2 diffuse .6 metallic specular 1. roughness .001 reflection .0}
#declare glass = finish {ambient .05 diffuse .3 specular 1. roughness .001}
#declare glass2 = finish {ambient .0 diffuse .3 specular 1. reflection .25 roughness .001}
#declare Rcell = 0.050;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
      torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
      translate LOC}
#end

atom(< -2.94,  -4.30,  -1.01>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #0 
atom(< -1.76,  -2.58,  -2.41>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #1 
atom(< -2.94,  -1.44,  -0.50>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #2 
atom(< -1.76,   0.28,  -1.91>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #3 
atom(< -2.94,   1.41,   0.00>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #4 
atom(< -1.76,   3.13,  -1.40>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #5 
atom(< -3.44,  -3.80,  -3.82>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #6 
atom(< -2.27,  -2.08,  -5.22>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #7 
atom(< -3.44,  -0.95,  -3.32>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #8 
atom(< -2.27,   0.77,  -4.72>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #9 
atom(< -3.44,   1.91,  -2.81>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #10 
atom(< -2.27,   3.63,  -4.22>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #11 
atom(< -3.95,  -3.31,  -6.63>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #12 
atom(< -2.77,  -1.59,  -8.03>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #13 
atom(< -3.95,  -0.45,  -6.13>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #14 
atom(< -2.77,   1.27,  -7.53>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #15 
atom(< -3.95,   2.40,  -5.63>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #16 
atom(< -2.77,   4.12,  -7.03>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #17 
atom(< -0.08,  -4.21,  -1.50>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #18 
atom(<  1.09,  -2.49,  -2.91>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #19 
atom(< -0.08,  -1.36,  -1.00>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #20 
atom(<  1.09,   0.36,  -2.40>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #21 
atom(< -0.08,   1.50,  -0.50>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #22 
atom(<  1.09,   3.22,  -1.90>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #23 
atom(< -0.59,  -3.72,  -4.32>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #24 
atom(<  0.59,  -2.00,  -5.72>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #25 
atom(< -0.59,  -0.86,  -3.81>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #26 
atom(<  0.59,   0.86,  -5.21>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #27 
atom(< -0.59,   2.00,  -3.31>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #28 
atom(<  0.59,   3.72,  -4.71>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #29 
atom(< -1.09,  -3.22,  -7.13>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #30 
atom(<  0.08,  -1.50,  -8.53>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #31 
atom(< -1.09,  -0.36,  -6.62>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #32 
atom(<  0.08,   1.36,  -8.03>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #33 
atom(< -1.09,   2.49,  -6.12>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #34 
atom(<  0.08,   4.21,  -7.52>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #35 
atom(<  2.77,  -4.12,  -2.00>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #36 
atom(<  3.95,  -2.40,  -3.40>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #37 
atom(<  2.77,  -1.27,  -1.50>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #38 
atom(<  3.95,   0.45,  -2.90>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #39 
atom(<  2.77,   1.59,  -0.99>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #40 
atom(<  3.95,   3.31,  -2.39>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #41 
atom(<  2.27,  -3.63,  -4.81>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #42 
atom(<  3.44,  -1.91,  -6.21>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #43 
atom(<  2.27,  -0.77,  -4.31>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #44 
atom(<  3.44,   0.95,  -5.71>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #45 
atom(<  2.27,   2.08,  -3.80>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #46 
atom(<  3.44,   3.80,  -5.21>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #47 
atom(<  1.76,  -3.13,  -7.62>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #48 
atom(<  2.94,  -1.41,  -9.03>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #49 
atom(<  1.76,  -0.28,  -7.12>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #50 
atom(<  2.94,   1.44,  -8.52>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #51 
atom(<  1.76,   2.58,  -6.62>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #52 
atom(<  2.94,   4.30,  -8.02>, 1.32, rgb <0.88, 0.40, 0.20>, 0.0, ase3) // #53 
