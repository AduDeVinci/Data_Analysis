from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("apply_Model", views.index3, name="index"),



]