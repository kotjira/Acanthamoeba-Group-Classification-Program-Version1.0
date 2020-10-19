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


PhotoIconCassify = PhotoImage(file="photo/icon_cassify2.png")
ButtonClassify = Button(frame1,image=PhotoIconCassify,borderwidth=0)
ButtonClassify.configure(background='#212121')
#ButtonClassify = Button(frame1,text = 'Classify', bg = 'white', fg = 'black',width=32,height=8)
ButtonClassify.grid(column=0, row=0, padx=4,pady=4)


PhotoIconTraining = PhotoImage(file="photo/icon_training2.png")
ButtonTraining = Button(frame1,image=PhotoIconTraining,borderwidth=0)
ButtonTraining.configure(background='#212121')
#ButtonTraining = Button(frame1, text = 'Training', bg = 'white', fg = 'black',width=32,height=8)
ButtonTraining.grid(column=1, row=0 ,padx=4,pady=4)


PhotoIconExit = PhotoImage(file="photo/icon_exit2.png")
ButtonExit = Button(frame1,image=PhotoIconExit,borderwidth=0,command=WindowMain.destroy)
ButtonExit.configure(background='#212121')
#ButtonExit = Button(frame1, text = 'Exit', bg = 'white', fg = 'black',width=67,height=3)
ButtonExit.grid(columnspan=2, row=1,padx=4,pady=0)



WindowMain.mainloop()