from django.shortcuts import render
from django.http import JsonResponse

import joblib

joblib.load('modelPipeline.pkl')

# Create your views here.

def scoreJson(request):
    return JsonResponse({'score' : 1})


def scorefile(request):
    return JsonResponse({'score' : 1})