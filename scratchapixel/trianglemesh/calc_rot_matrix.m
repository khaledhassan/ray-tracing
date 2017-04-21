## Copyright (C) 2017 Khaled Hassan
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} calc_rot_matrix (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Khaled Hassan <khaled@khaled-HP-EliteBook-840-G1>
## Created: 2017-04-16

function [retval] = calc_rot_matrix (xdeg, ydeg, zdeg)

  rot_x = [
           [1 0 0]; 
           [0 cos(xdeg * pi/180) -sin(xdeg * pi/180)]; 
           [0 sin(xdeg * pi/180) cos(xdeg * pi/180)]
          ];
          
  rot_y = [
           [cos(ydeg * pi/180) 0 sin(ydeg * pi/180)]; 
           [0 1 0]; 
           [-sin(ydeg * pi/180) 0 cos(ydeg * pi/180)]
          ];
          
  rot_z = [
           [cos(zdeg * pi/180) -sin(zdeg * pi/180) 0]; 
           [sin(zdeg * pi/180) cos(zdeg * pi/180) 0]; 
           [0 0 1]
          ];
          
  retval = rot_x * rot_y * rot_z;
endfunction
