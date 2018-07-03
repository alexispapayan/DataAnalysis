# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:16:51 2018

@author: papagian
"""
from Triangulation_with_points import *
#from Triangulation import *


contour=generate_contour(15)
#contour=read_contour('contour_30')
quality,_=quality_matrix(contour)
ordered_matrix=order_quality_matrix(quality,contour,check_for_equal=True)
triangulate(contour,ordered_matrix,recursive=True)
plot_contour(contour)
