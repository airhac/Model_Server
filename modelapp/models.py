from django.db import models
# Create your models here.
from django.core.validators import FileExtensionValidator

class Server_Model(models.Model):
    image_num = models.IntegerField(default=0)
    video = models.FileField(upload_to='video/', validators=[FileExtensionValidator(
                            allowed_extensions=['avi', 'mp4', 'mkv', 'mpeg', 'webm'])],
                            null=True,
                            blank=True)
    # 생성일자
    created_at = models.DateField(auto_now_add=True, null=True)