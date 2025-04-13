from django.urls import path
from Djangochef.views import Home

urlpatterns = [
    path("", Home.as_view() , name = 'home'),
]
