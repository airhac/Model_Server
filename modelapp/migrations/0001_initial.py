# Generated by Django 3.2.5 on 2021-11-18 02:04

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Server_Model',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image_num', models.IntegerField(default=0)),
                ('video', models.FileField(blank=True, null=True, upload_to='video/', validators=[django.core.validators.FileExtensionValidator(allowed_extensions=['avi', 'mp4', 'mkv', 'mpeg', 'webm'])])),
                ('created_at', models.DateField(auto_now_add=True, null=True)),
            ],
        ),
    ]