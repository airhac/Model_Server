from django import forms
from django.forms import ModelForm

from modelapp.models import Server_Model

class ServerForm(ModelForm):
    class Meta:
        model = Server_Model
        fields = ['image_num','video']

