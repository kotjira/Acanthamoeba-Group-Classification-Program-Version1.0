from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from skimage import feature
import PySimpleGUI as sg
import os.path
import PIL.Image
import io
import base64
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imutils
import glob
import h5py

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
exit_x_base64 = b'iVBORw0KGgoAAAANSUhEUgAAAGQAAAAiCAYAAACp43wlAAAACXBIWXMAAAsSAAALEgHS3X78AAAEjklEQVRoge2aTUhjVxTH/ye+oBCjXbSaIFa084rURQtmBgoW0pl2oUZBqLp0oSMU6gdxcK/gQmQWXRWKyJQiQgUXGgyUTpJFUbQV2wYF37RFxGCQKho1bczLO10khIpO4b48Iyb5wVvkcs8H5+S+e+67ByhQoECBewMZqayxsdFss9kem0ymTwA4iOgdAG8YbeeOYACnAP5k5nVN034Ih8O+jY2NuJFGDAmU0+ksKSsr6wMwBOBtZjYTUS4k4UaYmYkoDmAPwJeRSGQ6EAj8Y4TujIPmcrkeEdFXRPQ+gCIDfLpvJJj5V2b+3OPxrGeqLKOEtLe3dwL4GsnXUr5zAqB/cXFxPhMluv/RLpfrMyL6BoA1EwdyiBIArbIsK4qibOtVomuFtLS0fCBJkh+FlXETJ6qqfry8vPyLHmHhhDQ3NxebzeYfATj0GMwTfo7H401erzcmKmgSFZAkqYeZG0Xl8glmbpQkqUePrFBCnE6nBOBZLpe0RpCKz7NUvIQQSkhpaWkTgAeiRvKUB6l4CSGUQZPJ5EIWT93V1dUYHh7G0dFReszv92N1dfXG+Q6HA11dXRgbG8PFxQUGBgbg8/mwtbWVLZfTpFZJK4CAiJzoknokOD8jiouLUVNTg8nJyfTY+fn5a+cfHx9jc3MTsVhyL21oaMDa2tqt+/k6iOhDURnRTV0WNWAEh4eH6ScajaK+vh7j4+OoqKgAALS1tcHtdqOoqAiyLENVVYyMjKC8vBwdHR0YHR2FJAm/zo1A+PUu5CUzv5nt/ZyI0Nvbm/49NzcHRVFwfHwMt9uNxcVFdHZ2YmpqChaLBbKc/M/Mzs6ivr4efr8fwWAQqqpm1e8Uwuc04bL3LlhfX08/qqpC0zRMT0/DYrFgaGgI8/PzCAaDV2TC4TASiQROTk5wcHBwR56LI7RCiOgvALZb8uVGmPlasAHg8vISkUgENpsNkUgkmy6JcCIqILpCXokauC26u7tRUlKCmZkZ9PT0wG63X5uTSCRgtVphtd7Z57bfRQVEd7qfAHwkakQvqqoiGo1icHAwPbaysoJYLIampiZMTExgb28PtbW16Ovrw8LCwpUqLBAIoLW1FQ6H40qlli2Y+eb6/H8Q2qFdLpeTiHzZOqmbzWZUV1dfGTs6OoLJZILFYsH+/j4AwGKxoLKyEqFQCHa7Hbu7u+n5dXV10DTtylg24CSPPR5PQEROKLBOp1OyWq3bRHQn5e99gplfnZ2dvRcIBITKO6E9JKX8OTOzkHd5Rio+z0WTAegoe1VVfUFEG6Jy+QQRbaiq+kKPrHBCvF5vLB6P9yPZgVHgOqfxeLxfz10IoPNg6PV6NwE8BRDVI5/D/A3gaSo+utB9p76zs7Mty/IfRPQpkvfJ+c6ppml9S0tL32WiJKO2HUVRtmRZ9hHRQwBv4Z58ijGYBIDfNE3r9ng832eqLOM+KkVRQrIsfwvgkIjeBVDKzKZcvlX8T6PcLjOPMfMXHo9n1wjdhreSVlVVPWHmJwAeElEdgDLkRgNdAkAEySSsEdHLUCj00uhW0gIFChS4R/wL3bivrM9w6WcAAAAASUVORK5CYII='
home_x_base64 = b'iVBORw0KGgoAAAANSUhEUgAAAGQAAAAiCAYAAACp43wlAAAACXBIWXMAAAsSAAALEgHS3X78AAAFJklEQVRoge2a32tTZxjHP0+aNLlo2iJrrcROzEzB3WxSI4w6Tc2KqKFXK14KWgPDQUX8B3YreDEQpnNgZ0FhAy80mDoWV6RS1rV2bZlg7GosqUgZxaa/ZnJ63l2knlkmzDc/1Lb5wCGck/N9nvc8T573R84LJUqUKLFqkEIaa2xsdNTV1e2z2WyfATtF5AOgutB+3hIKmAHGlVL9pmn+/PTp09uDg4OZQjopSKACgYCrsrKyHegA3ldKOURkLSThlSillIhkgAng61Qq9V1PT8/fhbCdd9BCodAuEflGRD4CygrQptXGklJqWCn1RSQS6c/XWF4JaW1tbQO+JdstrXeeAeHr16//mI+RnH/RoVDocxH5HnDn04A1hAs45PP54vF4/H6uRnKqkIMHD35st9t/oVQZr+KZYRjNN2/e/D0XsXZCDhw44HQ4HL3AzlwcrhMGMpnM7mg0+lxXaNMV2O32I0qpRl3dekIp1Wi324/kotVKSCAQsAOn1/KUthAsx+f0cry00EpIRUXFbmCbrpN1yrbleGmhlRCbzRYqRnXU19dz5syZFdeCwSDt7e2FdvXGkCyHdHW6Y8guXQevg9PpxOv1rrhWXV3Npk2biuHujSEin+hqdPs4n66DQlBZWcnhw4fZuHEjdrude/fu0d3dTTqd5tSpUywsLFBTU4PNZiMWi+H3+3G5XBiGwYULF0ilUgA0NTWxd+9eTNMknU7T2dnJ9PR0MZuu3b1rVYhS6j1dB6+LiHDs2DHraGz8dyJ34sQJtmzZwuXLl7ly5Qr79++ntbUVgNraWnbs2MHVq1cZGBjg5MmTzM7O0tXVRXV1NW1tbQBs376dcDhMb28vXV1dlJWVEQ6Hi/U4L9Bep2lPe4tJf3+/dUxOTgLgdrvx+/1cunSJiYkJ4vE4165dIxgMWrpYLMbY2Bh37tzB4XDQ3d1NMpmkr68Pny9b1E1NTRiGgdfrpaWlBcgm6V1Dq8sSkb+AumI0RCnF6Oiodd7Q0MCGDRtwOByICDMzM9Z3qVSK8vJy63xubg6ATCaz4tM0Tes+l8vFkydPGBoasnS3bt0qxqO8zDNdgW6FPNR1kC/T09M8evSIPXv2UFZWhsvlorm5mZGRES07o6Oj1NfXMzc3x/DwMI8fP2ZqaqpIrbYY0xXoDuq/AZ/qOvk/DMOwBt4XpNNp5ufnATh37hxHjx5l69atuFwuAC5evAjA/Pz8iopIpVIsLS0B2UpZWFgA4O7duzQ0NNDR0cHY2Bi1tbUkk0nOnz9f6MexUEr16Wq01hShUCggIrcLvRZxOBx4PB4SiYR1raqqCqfTaf2KnU4nHo8HgPHxceu+uro6FhcXrS7N6/WSSCQwTZOqqircbjfJZNK6f/PmzZSXl2Oa5gp/hUZl2ReJRHp0dFqBDQQCdrfbfV9E3sr0dzWhlHo4Ozv7YU9Pj6Gj0xpDlo2fVUoprdatM5bjc1Y3GZDDtNcwjE4RGdTVrSdEZNAwjM5ctNoJiUajzzOZTJjsDowS/2Umk8mEc3kXAjkuDKPR6BBwHFjIRb+GWQSOL8cnJ3J+p/7gwYP7Pp/vTxFpIfs+eb0zY5pm+40bN37Ix0he23bi8fgfPp/vtoj4gRresb9i3hBLwIhpmocjkchP+RrLex9VPB6f9Pl8XcCUiDQAFUop21p+q/jSRrmEUuorpdSXkUgkUQjbBd9K6vF4gkqpIOAXES9QydrYQLcEpMgm4VcRiU1OTsYKvZW0RIkSJVYR/wCKqum80BwXygAAAABJRU5ErkJggg=='

#-----------------------------------ขนาดภาพ-----------------------------
def convert_to_bytes(file_or_bytes, resize=None):
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()
#-----------------------------------ขนาดภาพ-----------------------------

#-----------------------------------Colum Classify-----------------------------
left_col_classify = [    [sg.Text('Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER_CLASSIFY-'), sg.FolderBrowse()],
                [sg.Listbox(values=[], enable_events=True, size=(40,20),key='-FILE LIST_CLASSIFY-')],
                [sg.Text('Resize to'), sg.In(key='-W_CLASSIFY-', size=(5,1)), sg.In(key='-H_CLASSIFY-', size=(5,1))]
           ]

images_col_classify = [  [sg.Text('You choose from the list:')],
                [sg.Text(size=(40,1), key='-TOUT_CLASSIFY-')],
                [sg.Image(key='-IMAGE_CLASSIFY-')],
                [sg.Text('Group : '),sg.Text('1',background_color='#4e4e4e')] 
             ]
#-----------------------------------Colum Classify-----------------------------

#-----------------------------------Colum Training-----------------------------
left_col_training = [   [sg.Text('Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER_TRAINING-'), sg.FolderBrowse()],
                [sg.Listbox(values=[], enable_events=True, size=(40,20),key='-FILE LIST_TRAINING-')],
                [sg.Text('Resize to'), sg.In(key='-W_TRAINING-', size=(5,1)), sg.In(key='-H_TRAINING-', size=(5,1))]
            ]

images_col_training = [ [sg.Text('You choose from the list:')],
                [sg.Text(size=(40,1), key='-TOUT_TRAINING-')],
                [sg.Image(key='-IMAGE_TRAINING-')],
                [sg.Text('Group : '), sg.InputCombo(('Group 1','Group 2','Group 3'),key='-COMBO_GROUP_TRAINING-', size=(20, 1))] 
              ]
#-----------------------------------Colum Training-----------------------------

#-----------------------------------MAIN-----------------------------
classify_layout = [ [sg.Column(left_col_classify, element_justification='c'), sg.VSeperator(),sg.Column(images_col_classify, element_justification='c'),],
                    [sg.Button('', image_data=classify_x_base64,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-Classify-')]
                  ]

training_layout =[  [sg.Column(left_col_training, element_justification='c'), sg.VSeperator(),sg.Column(images_col_training, element_justification='c'),],
                    [sg.Button('', image_data=training_x_base64,button_color=(sg.theme_background_color(),sg.theme_background_color()),border_width=0, key='-Training-')]
                 ]

layout = [[sg.TabGroup([[sg.Tab('Classify', classify_layout), sg.Tab('Training', training_layout)]])]]    


window = sg.Window('Acanthamoeba Group Classification Program', layout,resizable=True)


while True:
    #------------------------------------ Classify-----------------------------
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == '-FOLDER_CLASSIFY-':                        
        folder = values['-FOLDER_CLASSIFY-']
        try:
            file_list = os.listdir(folder)         
        except:
            file_list = []
        fnames = [f for f in file_list if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith((".png", ".jpg", "jpeg", ".tiff", ".bmp"))]
        window['-FILE LIST_CLASSIFY-'].update(fnames)
    elif event == '-FILE LIST_CLASSIFY-':    
        try:
            filename = os.path.join(values['-FOLDER_CLASSIFY-'], values['-FILE LIST_CLASSIFY-'][0])
            window['-TOUT_CLASSIFY-'].update(filename)
            if values['-W_CLASSIFY-'] and values['-H_CLASSIFY-']:
                new_size = int(values['-W_CLASSIFY-']), int(values['-H_CLASSIFY-'])
            else:
                new_size = None
            window['-IMAGE_CLASSIFY-'].update(data=convert_to_bytes(filename, resize=new_size))
        except Exception as E:
            print(f'** Error {E} **')
            pass
    #------------------------------------ Classify-----------------------------

    #------------------------------------ Training-----------------------------
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == '-FOLDER_TRAINING-':                        
        folder = values['-FOLDER_TRAINING-']
        try:
            file_list = os.listdir(folder)         
        except:
            file_list = []
        fnames = [f for f in file_list if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith((".png", ".jpg", "jpeg", ".tiff", ".bmp"))]
        window['-FILE LIST_TRAINING-'].update(fnames)
    elif event == '-FILE LIST_TRAINING-':    
        try:
            filename = os.path.join(values['-FOLDER_TRAINING-'], values['-FILE LIST_TRAINING-'][0])
            window['-TOUT_TRAINING-'].update(filename)
            if values['-W_TRAINING-'] and values['-H_TRAINING-']:
                new_size = int(values['-W_TRAINING-']), int(values['-H_TRAINING-'])
            else:
                new_size = None
            window['-IMAGE_TRAINING-'].update(data=convert_to_bytes(filename, resize=new_size))
        except Exception as E:
            print(f'** Error {E} **')
            pass
    #------------------------------------ Training-----------------------------
    if event == '-Classify-':
        img = cv2.imread('-FILE LIST_CLASSIFY-')
        size = cv2.resize(img,None,None,0.6,0.6)
        gray = cv2.cvtColor(size,cv2.COLOR_RGB2GRAY)
        gaussian = cv2.GaussianBlur(gray,(5,5),0)
        lapla = cv2.Laplacian(gaussian,cv2.CV_64F)
        gauss = cv2.adaptiveThreshold(gaussian,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                 cv2.THRESH_BINARY,11,5)
        canny = cv2.Canny(gaussian,50,40)

        cv2.imshow('output',gauss)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #sg.popup_yes_no('Are you sure?') 
        
    if event == '-Training-':
        sg.popup_yes_no('Are you sure?') 
        #sg.popup('Are You srue?','The picture you slected is in the ', "COMBO_GROUP_TRAINING")


values = window.read()
window.close()

sg.SystemTray.notify('Successful', 'identification of Acanthamiba cyst')




