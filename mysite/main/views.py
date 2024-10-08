from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import generics
from .serializers import RequestSerializer

def index(request):
    return render(request, 'main/index.html')

def about(request):
    return render(request, 'main/about.html')

class modelAPIView(generics.ListAPIView):
    queryset = 'saddsd'
    serializer_class = RequestSerializer