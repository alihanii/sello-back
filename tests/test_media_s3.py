import boto3
import pytest
from moto import mock_aws

from app.core.config import settings
from app.services import media


@pytest.fixture
def s3(monkeypatch):
    with mock_aws():
        for k, v in {"storage": "s3", "s3_bucket": "sello", "s3_region": "us-east-1", "s3_access_key": "x",
                     "s3_secret_key": "y", "s3_endpoint_url": "", "s3_prefix": "prod", "s3_public_url": ""}.items():
            monkeypatch.setattr(settings, k, v)
        media._s3.cache_clear()
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="sello")
        yield
    media._s3.cache_clear()


def test_save_and_read_from_s3(s3):
    url = media.save_bytes(7, b"img", "a.png", "image/png")
    assert url.startswith("/media/7/") and url.endswith(".png")
    assert media.read_local(url) == (b"img", "image/png")
    key = "prod/" + url.removeprefix("/media/")
    assert boto3.client("s3", region_name="us-east-1").get_object(Bucket="sello", Key=key)["Body"].read() == b"img"
    assert media.read_local("/media/7/missing.jpg") is None
    assert media.read_local("/media/../etc/passwd") is None
    assert media.read_local("https://example.com/a.jpg") is None


def test_public_url_signed_or_public(s3, monkeypatch):
    url = media.save_bytes(1, b"x", content_type="image/jpeg")
    signed = media.public_url(url)
    assert "prod/1/" in signed and "Signature" in signed
    monkeypatch.setattr(settings, "s3_public_url", "https://cdn.example.com/")
    assert media.public_url(url) == "https://cdn.example.com/prod/" + url.removeprefix("/media/")


def test_local_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "media_dir", str(tmp_path))
    url = media.save_bytes(3, b"abc", "x.webp")
    assert (tmp_path / url.removeprefix("/media/")).read_bytes() == b"abc"
    assert media.read_local(url) == (b"abc", "image/webp")
    assert media.public_url(url) == f"{settings.public_base_url.rstrip('/')}{url}"
