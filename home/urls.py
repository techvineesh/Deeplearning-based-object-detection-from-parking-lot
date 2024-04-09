from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('files', views.get_files, name='get_file'),
]