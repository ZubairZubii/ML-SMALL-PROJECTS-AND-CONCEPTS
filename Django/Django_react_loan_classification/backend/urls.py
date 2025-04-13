
from django.contrib import admin
from django.urls import path
from firstapp.views import views
from django.views.decorators.csrf import csrf_exempt


urlpatterns = [
    path("admin/", admin.site.urls),
     path('scorejson', csrf_exempt(views.scoreJson), name='scoreApplication'),

    path('scorefile' ,csrf_exempt(views.scorefile ), name='scorefile')
]
