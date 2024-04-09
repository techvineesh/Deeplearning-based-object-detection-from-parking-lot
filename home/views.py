from django.shortcuts import render, redirect
from django.http import HttpResponse
from detecto.core import Model
from shutil import copyfile
import show
import os

# Create your views here.
def index(request):
    print(os.getcwd())
    return render(request, 'home/index.html')

def get_files(request):
    if request.method == 'POST':
        video = request.FILES["upload"]
        file = open("video.mp4", 'wb')
        file.write(video.read())
        file.close()
        print("Working")
        model = Model.load('Objmodel1.h5', ['occupied', 'unoccupied'])
        show.detect(model, 'video.mp4', 'output.avi')
        copyfile('output.avi', 'home/static/home/video/output.avi')
        return redirect('/static/home/video/output.avi')
    return HttpResponse('media/output.mp4')
