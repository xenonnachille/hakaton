from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import generics
from .serializers import RequestSerializer
from . import fitted_model
import json 


def foo(arguments):
    # Логика, которую хотите выполнить
    return fitted_model.predict([arguments['question'], arguments['answer']])

def index(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        answer = request.POST.get('answer')
        
        # Создаем аргументы для функции foo
        arguments = {'question': question, 'answer': answer}
        
        # Вызываем функцию foo
        result = foo(arguments)
        return render(request, 'main/index.html', {'result': result})
    data = dict(request.GET)
    if 'fit' in data.keys():
        arguments = json.loads(str(data['fit'][0]))
        return HttpResponse(fitted_model.predict([arguments['question'], arguments['answer']]))
    return render(request, 'main/index.html')

def about(request):
    return render(request, 'main/about.html')

class modelAPIView(generics.ListAPIView):
    queryset = 'saddsd'
    serializer_class = RequestSerializer