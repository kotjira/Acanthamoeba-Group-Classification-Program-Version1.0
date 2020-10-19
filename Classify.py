from tkinter import *
WindowMain = Tk()
WindowMain.geometry("1920x1080")
WindowMain.configure(background='#212121')
WindowMain.title('Acanthamoeba Group Classification Program')
WindowMain.option_add('*font','tahoma 20')

frame1 = Frame(WindowMain)
frame1.place(width=1920, height=1080, x=0, y=0)
frame1.option_add('*font','tahoma 12')
frame1.configure(background='#212121')
frame1.place(x=450,y=220)



WindowMain.mainloop()