from rest_framework import serializers


class Request:
    def __init__(self, content):
        self.content = content

class RequestSerializer(serializers.Serializer):
     email = serializers.CharField()

   