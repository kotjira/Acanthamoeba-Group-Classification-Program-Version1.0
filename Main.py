import PySimpleGUI as sg
import cv2
import warnings
import joblib
import os
import os, io
import os.path
import PIL.Image
import io
import base64
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imutils
import glob
import h5py
import sys 
from PyQt5.QtWidgets import *
from PyQt5 import QtCore 
from PyQt5 import QtGui
from PIL import Image
from matplotlib import pyplot
from matplotlib import pyplot as plt
from skimage import feature
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#---------------------------------Theme GUI---------------------------------
sg.LOOK_AND_FEEL_TABLE['Theme'] = {'BACKGROUND': '#212121',
                                        'TEXT': '#aaaaaa',
                                        'INPUT': '#ffffff',
                                        'TEXT_INPUT': '#000000',
                                        'SCROLL': '#ffffff',
                                        'BUTTON': ('white', '#4e4e4e'),
                                        'PROGRESS': ('#01826B', '#D0D0D0'),
                                        'BORDER': 0, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0,
                                        }
sg.theme('Theme')



ok_x_base64 = b'iVBORw0KGgoAAAANSUhEUgAAAGQAAAAiCAYAAACp43wlAAAACXBIWXMAAAsSAAALEgHS3X78AAAEi0lEQVRoge2az0sjZxjHv086WQOJsStWRKXgj1EpQoPZFSsrpt32oAZhsaVHD+4PCqIXPemlf8AeeipoD/4AD62CaHBQmm0OhdptpaZLFWe7YRHUbPWQjaK1mZmnB0NourbwzqSxmnyO77zf53l4nnln3pn3AfLkyZPn0kCZNOb1eu1lZWXv2Wy29wHcIKIaAK9n2s8FwQBeAogw82PDML6ORqOP1tbWEpl0kpFE+Xw+h9vtvgtgEMCbzGwnoqtQhHNhZiaiBIBtAJ/F4/EvQqHQ75mwbTlpfr+/mYg+J6K3AbyWgZguGzozh5n5k0Ag8NiqMUsF6e7u/gjAGM4eS7lODMD9hYWFr6wYMX1H+/3+D4loEkChlQCuEA4AXbIsq6qqbpg1YmqFdHZ2eiRJ+gb5lXEeMU3T3l1aWlo3IxYuSEdHR4Hdbv8WwA0zDnOEHxOJxC1FUU5FhTZRgSRJvczsFdXlEszslSSp14xWqCA+n08CMHSVt7SZIJmfoWS+hBAqiMvlugWgVtRJjlKbzJcQQhW02Wx+XMBXt8PhQG1tLRwOBzRNQyQSQTweT12vqanB0dERXrx4AQCorKyE2+3GxobpzY5lkqukC0BIRCf6DmkWnG+Z4uJijIyMoK2tDaenp/B4PBgZGUFDQ0NqTk9PD1paWgAAsixjdHQUzc1ZD/UViOgdUY1oQWRRB1Z58OAB9vf3MTY2hidPnmBiYgLBYBADAwO4du1a2tyGhgYMDw9jZWUF09PT2Q71PIQf70IFYeYSUQdWcDqd8Hq9mJ2dha7rqfFgMIiCgoK0VVJfX4+hoSEoioL5+fm0+ReI8Hea8C4gmxQXF0PXdezu7qaN67qOaDSK0tLS1JjH4wEzY3NzE4ZhZDvUjCG0Qojo4L8K5DxOTk4gSRKKior+HgdKSkpweHiYGlteXsbMzAwGBgZQXl6ezTD/jZioQPQd8lTUgRUODg6wvb2Nrq6utPGmpia4XC5sbW2lxmKxGBRFwfr6OgYHB+F0OrMZ6j/xq6hAtCA/iDqwyvj4ONrb23Hnzh1UV1ejtbUVfX19mJqaQiyWfgMahoHJyUkcHR2hv78fdrs92+GmwczfiWqE/vbKsvwHgN5sfqnv7+8jHA6jqqoKjY2NuH79Oubm5rC6upqa43Q6sbe3h2g0Cl3XEQ6HUVdXh0gkguPj42yFmgafMaqq6nMRnVBifT6fVFhYuEFEWd/+XjaY+enh4eFboVBIE9EJPbKSxh8yMwtFl2Mk8/NQtBiAib+9mqZNENGaqC6XIKI1TdMmzGiFC6IoymkikbiPsw6MPK/yMpFI3DdzFgKYKAgAKIryE4B7AC7mjfn/5QTAvWR+TGH6TH1ra2tDluVnRPQBzs6Tc52XhmHcXVxc/NKKEUttO6qq/iLL8iMiugngDZhccZccHcDPhmF8HAgEVqwas9xHparqjizL0wB+I6I6AC5mtl3lU8W/NMo9Z+ZPmbk/EAg8z4TtjLeSVlRU3Gbm2wBuElE1ADeuRgOdDiCOsyJ8T0TBnZ2dYKZbSfPkyZPnEvEn2DGbON5q8DcAAAAASUVORK5CYII='
cancle_x_base64 = b'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'
back_x_base64 = b'iVBORw0KGgoAAAANSUhEUgAAAGQAAAAiCAYAAACp43wlAAAACXBIWXMAAAsSAAALEgHS3X78AAAFJklEQVRoge2a32uTVxjHPydNaEibtrFNFaMTOlOxXnQxtTBwkOmGqNEbNwpetChWLYwOtP/AwFsvdmOhrGDphbLhhRobGKsNtVDiVvZDVmjSrWEatclq2qRsDXmTs4uGItMNzpsYaZvPZZLv9znv8+T8es+BMmXKlFk3iGKaud1u07Zt2w4ZDIaPgDYhxLtAXbHjvCUksAT8LqV8mMvlvnv+/Pn9qampTDGDFCVRHo/HXFNTcw74HHhHSmkSQmyEIrwWKaUUQmSAP4Avk8nkV4FAYKUY3gUnzev1tgsh+oUQrUBFEdq03shKKX+WUvb4fL6HhZoVVJCTJ09+CgywOixtdhaB83fu3PmmEBPd/2iv1/uJEGIIsBbSgA2EGTjudDpDoVBoWq+Jrh5y7Nix94xG4xjlnvE6FjVN+3BkZOQnPWLlghw9erTSZDJNAG16Am4SfshkMgf9fn9aVWhQFRiNxi4ppVtVt5mQUrqNRmOXHq1SQTwejxHo28hL2mKQz09fPl9KKBWkurr6ILBbNcjbpKWlha1bt76N0Lvz+VJCqYIGg8FLCXbdp06dorm5mXg8TkNDA8FgkLGxMV1enZ2djI+PMzIyUuRW/j/5XnIcCKjoVOeQdsXf62Lv3r3Mzc0xPDzMvXv3uHDhAjt27ChF6KIihHhfVaM6xjlVA+hF0zTS6TTz8/MYjUbMZjO7du2is7OTdDpNVVUVo6OjjI+PA2Cz2ejo6MBut5PNZllcXOTatWtrflarle7ubp49e8aNGzdK9RjKw7tSQaSUDaWaz/fv38+WLVtwuVyEw2Hm5uYwm83cunWLRCJBa2srFy9eJBgMkk6n6e3tBWBoaIhsNovFYlnzqq2t5fLly0gpGRwcLEn78yjv05RXAaUiGo0yOTkJgMvlYufOnTx+/Jja2lra29vZs2cPVVVVOBwO4vE4ra2tXLp0iUgk8orXiRMnSCaT9PX1sby8XOInUUNpDhFC/PmmGvJv5ufnefToEQMDA8RiMVwuFx6Ph56eHhKJBDdv3kTTNAwGAyaTCSHEfyY7HA5jsVhoayv5XnZRVaA6qYdVAxRKfX09drudlZUV9u3bx/T0NLdv30ZKicFgQAjBixcvePLkCUeOHMFisVBZWUl9ff2aRzAYZHBwkDNnzuB2l3RPO6sqUB2yvgc+UA2iSiqVwu12s337dux2O7Ozszx48IBEIsHp06fp7e2lsbGRWCxGTU0NAP39/XR1ddHU1ISmaVRUVHDlyhWePn1KMplkYmKCuro6zp49SyQSYWFh4U0/BlLKSVWN0gzt9Xo9Qoj7b3qnbrPZsNlsACwvLxOLxda+a2xspLq6moWFBaxWK0tLS6RSKQAqKytxOBwAxOPxtc9fpqmpiWg0Sjqt/JpJCbnKIZ/PF1DRKSXW4/EYrVbrtBCiZMvf9YqUMpxKpVoCgYCmolOaQ/LmV6WUUql1m4x8fq6qFgN0vO3VNO26EGJKVbeZEEJMaZp2XY9WuSB+vz+dyWTOs3oDo8yrLGUymfN6zkJAR0EA/H7/j0A38Jce/Qbmb6A7nx9d6D5Tn5mZmXY6nb8JIT5m9Tx5s7OUy+XO3b179+tCTAq6thMKhX51Op33hRAHADs6e9w6Jwv8ksvlOnw+37eFmhV8jyoUCkWdTucwEBNCNAPVUkrDRj5VfOmiXERK+YWU8jOfzxcphnfRr5I6HI7DUsrDwAEhRBNQw8a4QJcFkqwWISiEGI1Go6PFvkpapkyZMuuIfwAUw9vacZQWZwAAAABJRU5ErkJggg=='
classify_x_base64 = b'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'
training_x_base64 = b'iVBORw0KGgoAAAANSUhEUgAAAGQAAAAiCAYAAACp43wlAAAACXBIWXMAAAsSAAALEgHS3X78AAAFf0lEQVRoge2aX2hTVxzHPydNrNo0WLpqRax29RZfdIWmwsBBWjdEDeLD2iKIRdDCoLCH+upD8VUfBsJghMLMQ2FDFBsNjNVFEOa6SteVDUxdCZXWUGdsapLe5qb37CFt12rtOMnVaZvP27339/v+7vn97vlz7z1QoECBAu8Nwkqx+vp6R2VlZZPNZvsUcAshaoAtVsf5n5BAHBiVUvabpvljNBq98+DBA8PKIJYkyuPxbHS5XGeBL4EqKaVDCLEWirAiUkophDCAMeCr6elpXygU0q3QzjtpXq/3gBDiayHER0CRBff0vjEnpRySUn4RCAT68xXLqyDHjx9vBr4hOyytd6aA9ps3b36fj0jOT7TX6/1cCPEtUJrPDawhNgLHNE0Lh8PhP3MVyamHHD16tM5ut/9EoWesxFQmk2m8ffv2b7k4KxfkyJEjxQ6H4x7gziXgOmHAMIyDwWBwVtXRpupgt9vbpJT1qn7rCSllvd1ub8vFV6kgHo/HDpxfy0taK5jPz/n5fCmhVBCn03kQ2KMaZJ2yZz5fSihV0Gazeclzqexyuejo6MgGt9txOp1MTU0BMDQ0xK1bt/KRB8DtdtPS0kJXVxfJZDJvvVyY7yXHgJCKn2qXOqBo/wovXrzA5/MBsH//flpbW7l06RIAMzMz+coDEIvFGBwcZHZWeU61FCHEx6o+qgXRVAO8jJSSyclJAOLxOKZpLh4DdHZ2ous65eXl9PT0UF9fz969e0mn0+i6js/nY3p6mn379nHixAlSqRQOhwObzUZ3dzfRaJSioiI0TSOTyaxqZ7fbOXnyJLt378bpdKLrOhcvXiSTyeTbzAWUh3elOURK+YFqAFW2bduG2+2mp6eHx48fc//+fa5evYrf76eqqorDhw8DUFRUhNvtZmJiAr/fj2manDlzBoCSkhI0TftPO7fbjcfjwe/3Mzo6isvlsrIYkMN7mvKy901jmiZ9fX2MjIyg6zrJZJKGhgaam5txuVzU1tYu2uq6zrVr1xgfH+fu3btUV1evqPk6u61btzIxMUEkEmFgYIANGza8lTauhtKQJYT4G6h8Q/eySCwWA8DhcHDhwgWeP39Ob28vhmGwZcu/D106nSadTgOQyWTYtGnTinqvsxscHOTUqVO0t7dTXV1NKBSyuilTqg6qPWRENUA+lJWVsWvXLrq7uxkeHqasrAwrX4GePHmCEALDMLhx4wbXr1+3THueR6oOqpP6r8AnqkFex9zc3CvL0mQyiWFk//nEYjEGBgZoa2sjnU5TWlq6uHIyDINEIrFMa+F46bXV7DKZDD6fj6amJmpqakilUgwPD1vVPKSUP6v6KD1uXq/XI4S4Y9WbeklJCeXl5YyNjS2eq6ysZGZmhng8DmSHrZ07d2KaJk+fPl20Ly4uZvv27UQikUWtiooKIpHIsmur2S1QXFxMS0sLdXV1dHZ2WtE0ZJamQCAQUvFT6iGJROJeaWnpIyxY/kK2N7zcQ6LR6LJjwzAYHR1d5gMwOzu7LKlLtZZeW81u8+bNnD59mv7+fioqKkilUlY0a4FHiUTinqqT0hwSCoUywGUppVQN9C5is9l49uwZjY2N6LrOlStXLNGdz8/l+XwpUfj8/mZ4e5/fg8HgrGEY7WR3YBR4lbhhGO25FANyfDEMBoODwDnA0kF3DTADnJvPT07k/E/94cOHf2qa9pcQ4jOy/5PXO3HTNM/29vZ+l49IXtt2wuHwH5qm3RFCNAAVvIOfYt4Cc8Dvpmm2BgKBH/IVy3sfVTgcHtc0zQ9MCiFqAaeU0raW/you2SgXkVJ2SSk7AoFAxApty7eS7tix45CU8hDQIIT4EHCxNjbQzQHTZIvwixCib3x8vM/qraQFChQo8B7xD9P+fdq58aRQAAAAAElFTkSuQmCC'
exit_x_base64 = b'iVBORw0KGgoAAAANSUhEUgAAAFEAAAAiCAYAAAAnOTVZAAAACXBIWXMAAAsSAAALEgHS3X78AAAAIGNIUk0AAHolAACAgwAA+f8AAIDpAAB1MAAA6mAAADqYAAAXb5JfxUYAAAQ/SURBVHja7JpPSGNXFIe/G58ojdFZFI1IKtoJhLpoIelAwUKcKQPRKAhVl9LqCF34hzi4V3AhMotZFTpW3IgMgosYzKZjslK0iBaxC6cUsYoiozhR08a8vNvFZNKmSXRmWhKr7wcP3rvcB5fvnXPPOfc8IaVE17+TQUegQ7wSUl7fCCEunGi32/PNZvNdg8HwBeAQQnwI3ALENeAggZfAr1LKZU3Tftjf359fWVmJXvhSfCsUiZsMEJ1OZ2FxcXEn0At8IKXMF5cR/z/TlFIKIaLANvA4FAqNBYPBP94ZotvtviOE+FYI8TGQdwM9NSal/ElK+Y3P51t+a4hNTU0twHdxl73pOga6vF7vdDqIaQOL2+3+EpjQASZ0C5iIc0lRiiXW19d/oihKQAeY3iJVVa2bm5tby2iJLperQFGUJzrAzBapKMoTl8tVkDFPVBSlXUpp11ldGL3tiqK0p4XodDoV4OF1Tl/+C8X5PIzzSoZYVFRUC9zWMb2Rbsd5JVcsBoPBnc3qw2Kx0NfXx+HhYWIsEAiwuLiYdr7D4aC1tZXBwUHOzs7o7u5mfn6ejY2NXFljAxBMggjcyeZCCgoKqKysZGRkJDF2enqacf7R0RGrq6tEIhEAampqWFpayqVbf5YusFhzsZiDg4PEFQ6HsdlsDA0NUVpaCkBjYyMej4e8vDysViuqqtLf309JSQnNzc0MDAygKEpOXDrFnaWU72c7pggh6OjoSDxPTU2xubnJ0dERHo8Hr9dLS0sLo6OjGI1GrNZX33lychKbzUYgEGB9fR1VVXOVgGeuWLKp5eXlxKWqKpqmMTY2htFopLe3l+npadbX15Pe2d/fJxaLcXx8zN7eXs6jzN+Pwl4A5iznXCmAAM7PzwmFQpjNZkKh0FWup1Ms8flVWV1bWxuFhYWMj4/T3t5OeXl56vFKLIbJZMJkMuVqmb+kWCLwI/B5tlagqirhcJienp7E2MLCApFIhNraWoaHh9ne3qaqqorOzk5mZmaSoncwGKShoQGHw5EU4bPoRYspBxCNjY1OIcR8tiqW/Px8LBZL0tjh4SEGgwGj0cjOzg4ARqORsrIydnd3KS8vZ2trKzG/uroaTdOSxrIEUEop787OzgaTINbV1Skmk+lnIYQVXZdBfH5ycvJRIBBQk/bEYDCoAo+k3kO91AqBR3FeqSmOqqoTQogVHdWFue2KqqoTaU9xAPx+fyQajXbxqvOlK1Uvo9Fol9/vj2SEGAe5CjwAwjqzJP0OPIjz4UKIAF6vd1rTtK91i/zLAjVN++qfjapLyz6fz/dU07T7wBoQu6HwYsCapmn3fT7f04z75GXNe7fb/Z4QokMI0QtYblDz/jcp5WMp5fc+ny+cqWx9I4ivZbfb8ysqKu5JKe8BnwohqoFirkdTPwaEgC0p5ZIQ4tnu7u6zt/6NRNe7S/8rTId4NfTnAP872yfX0O5FAAAAAElFTkSuQmCC'
home_x_base64 = b'iVBORw0KGgoAAAANSUhEUgAAAGQAAAAiCAYAAACp43wlAAAACXBIWXMAAAsSAAALEgHS3X78AAAFJklEQVRoge2a32tTZxjHP0+aNLlo2iJrrcROzEzB3WxSI4w6Tc2KqKFXK14KWgPDQUX8B3YreDEQpnNgZ0FhAy80mDoWV6RS1rV2bZlg7GosqUgZxaa/ZnJ63l2knlkmzDc/1Lb5wCGck/N9nvc8T573R84LJUqUKLFqkEIaa2xsdNTV1e2z2WyfATtF5AOgutB+3hIKmAHGlVL9pmn+/PTp09uDg4OZQjopSKACgYCrsrKyHegA3ldKOURkLSThlSillIhkgAng61Qq9V1PT8/fhbCdd9BCodAuEflGRD4CygrQptXGklJqWCn1RSQS6c/XWF4JaW1tbQO+JdstrXeeAeHr16//mI+RnH/RoVDocxH5HnDn04A1hAs45PP54vF4/H6uRnKqkIMHD35st9t/oVQZr+KZYRjNN2/e/D0XsXZCDhw44HQ4HL3AzlwcrhMGMpnM7mg0+lxXaNMV2O32I0qpRl3dekIp1Wi324/kotVKSCAQsAOn1/KUthAsx+f0cry00EpIRUXFbmCbrpN1yrbleGmhlRCbzRYqRnXU19dz5syZFdeCwSDt7e2FdvXGkCyHdHW6Y8guXQevg9PpxOv1rrhWXV3Npk2biuHujSEin+hqdPs4n66DQlBZWcnhw4fZuHEjdrude/fu0d3dTTqd5tSpUywsLFBTU4PNZiMWi+H3+3G5XBiGwYULF0ilUgA0NTWxd+9eTNMknU7T2dnJ9PR0MZuu3b1rVYhS6j1dB6+LiHDs2DHraGz8dyJ34sQJtmzZwuXLl7ly5Qr79++ntbUVgNraWnbs2MHVq1cZGBjg5MmTzM7O0tXVRXV1NW1tbQBs376dcDhMb28vXV1dlJWVEQ6Hi/U4L9Bep2lPe4tJf3+/dUxOTgLgdrvx+/1cunSJiYkJ4vE4165dIxgMWrpYLMbY2Bh37tzB4XDQ3d1NMpmkr68Pny9b1E1NTRiGgdfrpaWlBcgm6V1Dq8sSkb+AumI0RCnF6Oiodd7Q0MCGDRtwOByICDMzM9Z3qVSK8vJy63xubg6ATCaz4tM0Tes+l8vFkydPGBoasnS3bt0qxqO8zDNdgW6FPNR1kC/T09M8evSIPXv2UFZWhsvlorm5mZGRES07o6Oj1NfXMzc3x/DwMI8fP2ZqaqpIrbYY0xXoDuq/AZ/qOvk/DMOwBt4XpNNp5ufnATh37hxHjx5l69atuFwuAC5evAjA/Pz8iopIpVIsLS0B2UpZWFgA4O7duzQ0NNDR0cHY2Bi1tbUkk0nOnz9f6MexUEr16Wq01hShUCggIrcLvRZxOBx4PB4SiYR1raqqCqfTaf2KnU4nHo8HgPHxceu+uro6FhcXrS7N6/WSSCQwTZOqqircbjfJZNK6f/PmzZSXl2Oa5gp/hUZl2ReJRHp0dFqBDQQCdrfbfV9E3sr0dzWhlHo4Ozv7YU9Pj6Gj0xpDlo2fVUoprdatM5bjc1Y3GZDDtNcwjE4RGdTVrSdEZNAwjM5ctNoJiUajzzOZTJjsDowS/2Umk8mEc3kXAjkuDKPR6BBwHFjIRb+GWQSOL8cnJ3J+p/7gwYP7Pp/vTxFpIfs+eb0zY5pm+40bN37Ix0he23bi8fgfPp/vtoj4gRresb9i3hBLwIhpmocjkchP+RrLex9VPB6f9Pl8XcCUiDQAFUop21p+q/jSRrmEUuorpdSXkUgkUQjbBd9K6vF4gkqpIOAXES9QydrYQLcEpMgm4VcRiU1OTsYKvZW0RIkSJVYR/wCKqum80BwXygAAAABJRU5ErkJggg=='
icon_x_base64 = b'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'

#---------------------------------Theme GUI---------------------------------


def get_image_as_data(filename, width=None, height=None):
    im = Image.open(filename)
    if isinstance(width, int) and isinstance(height, int): # Resize if dimensions provided
        im = im.resize((width, height))
    im_bytes = io.BytesIO()
    im.save(im_bytes, format="PNG")
    return im_bytes.getvalue()

def get_image_files_list(folder):
    all_files = os.listdir(folder)
    image_files = []
    for file in all_files:
        extension = file.lower().split(".")[-1]
        print(file, extension)
        if extension in ["jpg", "png", "jpeg", "jpe"]:
            image_files.append(file)
    image_files.sort()
    return image_files

def demo_photo_picker3(default_folder, default_pic):
    folder = default_folder
    files_listing = get_image_files_list(folder)
    column1 = [
        [
            sg.Listbox(values=files_listing,
                change_submits=True, # trigger an event whenever an item is selected 
                size=(40, 20), 
                font=("Arial", 10),
                key="files_listbox")
        ]
    ]
    column2 = [
        [
            sg.Image( data=get_image_as_data(default_pic, 340, 340), 
                key="image", size=(340,340))
        ] 
    ]

    layout = [
        [ 
            sg.Text("Select your images folder"),
            sg.InputText(key="photo_folder", change_submits=True), # trigger an event whenever the item is changed 
            sg.FolderBrowse(target="photo_folder")
        ], [
            sg.Column( column1 ), 
            sg.Column( column2 )
        ], [
            sg.Button('', image_data=classify_x_base64,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-Classify-'),
            sg.T(' '  * 115),
            sg.Button('', image_data=exit_x_base64,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-Exit-')
        ],
    ]

    window = sg.Window('Acanthamoeba Group Classification Program',layout,icon="GUI\icon_program.ico") #.Layout(layout)
    
    while True:
        event, values = window.Read()
        if event == None:
            break

        print(event)
        print(values)
        if event == "photo_folder":
            if values["photo_folder"] != "":
                if os.path.isdir(values["photo_folder"]):
                    folder = values["photo_folder"]
                    image_files = get_image_files_list(values["photo_folder"])
                    window.FindElement("files_listbox").Update(values=image_files)
                    if len(image_files) > 0:
                        full_filename = os.path.join(folder,image_files[0])
                        window.FindElement("image").Update(data=get_image_as_data(full_filename, 340, 340))
        if event == "files_listbox":
            full_filename = os.path.join(folder,values["files_listbox"][0])
            abp = os.path.relpath(full_filename)
            window.FindElement("image").Update(data=get_image_as_data(full_filename, 340, 340))
        if event == '-Classify-':
            warnings.filterwarnings('ignore')
            num_trees = 100
            test_size = 0.10
            seed      = 9
            train_path = "Dataset/Training"
            test_path  = "Dataset/Test"
            h5_data    = 'output/data.h5'
            h5_labels  = 'output/labels.h5'
            scoring    = "accuracy"

            train_labels = os.listdir(train_path)
            train_labels.sort()
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            models = []
            models.append(('LR', LogisticRegression(random_state=seed)))
            models.append(('LDA', LinearDiscriminantAnalysis()))
            models.append(('KNN', KNeighborsClassifier()))
            models.append(('CART', DecisionTreeClassifier(random_state=seed)))
            models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
            models.append(('NB', GaussianNB()))
            models.append(('SVM', SVC(random_state=seed)))

            results = []
            names   = []

            # import the feature vector and trained labels
            h5f_data  = h5py.File("h5_data.hdf5", 'r')
            h5f_label = h5py.File("h5_labels.hdf5", 'r')

            global_features_string = h5f_data['dataset_1']
            global_labels_string   = h5f_label['dataset_1']

            global_features = np.array(global_features_string)
            global_labels   = np.array(global_labels_string)

            h5f_data.close()
            h5f_label.close()

            (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                                np.array(global_labels),
                                                                                                test_size=test_size,
                                                                                                random_state=seed)
                                                                                                
            clf = SVC(random_state=seed)
            clf.fit(trainDataGlobal, trainLabelsGlobal)
            images_classify = cv2.imread(abp)
            fixed_size = tuple((100, 100))
            image_resize = cv2.resize(images_classify, fixed_size)
            featurehog = hog_feature(image_resize)
            global_feature = np.hstack([featurehog])
            scaler = MinMaxScaler(feature_range=(0, 1))
            rescaled_feature = scaler.fit_transform([global_feature])
            prediction = clf.predict(global_feature.reshape(1,-1))[0]
            ans = prediction_table(prediction)
            sg.popup("Your selected image is in group:"+ans,title = "Classify",icon = icon_x_base64)
    
            window.FindElement("image").Update(data=get_image_as_data(full_filename, 340, 340))
        
        if event is None or event == '-Exit-':
            return None
        
        if event is None or event == 'Exit':
            return None

    window.Close()

def hog_feature(image):
    #Grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Gaussain Blur Deleted noise
    gaussain = cv2.GaussianBlur(img,(5,5),0)
    # Canny Edge Detection
    canny_image = cv2.Canny(gaussain,100,10)
    hog_desc = feature.hog(canny_image, orientations=9, pixels_per_cell=(6, 6),
                cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    return hog_desc

def prediction_table(prediction_ans): 
    switcher = { 
        0: "1", 
        1: "2", 
        2: "3", 
    }
    return switcher.get(prediction_ans, "nothing") 

if __name__ == "__main__":
    default_pic = "GUI\Select.png"
    result = demo_photo_picker3(".", default_pic)
    print(result)