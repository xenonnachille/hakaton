from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import generics
from .serializers import RequestSerializer
from . import fitted_model
import json 


def foo(arguments):
    return fitted_model.get_result(arguments['question'], arguments['answer'])

def index(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        answer = request.POST.get('answer')
        arguments = {'question': question, 'answer': answer}
        result = foo(arguments)
        if result:
            result = 'Введенный вами ответ является релевантным для данного вопроса'
        else:
            result = 'Введенный вами ответ не является релевантным для данного вопроса'
        return render(request, 'main/index.html', {'result': result})
    data = dict(request.GET)
    if 'fit' in data.keys():
        arguments = json.loads(str(data['fit'][0]))
        return HttpResponse(fitted_model.get_result(arguments['question'], arguments['answer']))
    return render(request, 'main/index.html')

def about(request):
    return render(request, 'main/about.html')

class modelAPIView(generics.ListAPIView):
    queryset = 'saddsd'
    serializer_class = RequestSerializer