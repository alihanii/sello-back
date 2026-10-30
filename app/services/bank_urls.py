"""URL names az-iranian-bank-gateways reverses; served by FastAPI (app/routers/payments.py), not Django."""

from azbankgateways.urls import az_bank_gateways_urls
from django.urls import path

urlpatterns = [path("api/v1/payments/", az_bank_gateways_urls())]
