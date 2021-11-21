from django.urls import path
from modelapp.views import UserView

app_name = 'modelapp'

urlpatterns = [
    path('model/', UserView.as_view(), name='model'),
]