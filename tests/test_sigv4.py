import unittest

from oneharness.sigv4 import Credentials, now, sign_request, signing_key

# The vectors AWS publishes in its own documentation, so a passing test means
# "agrees with AWS", not "agrees with itself".
S3_CREDENTIALS = Credentials(
    access_key="AKIAIOSFODNN7EXAMPLE",
    secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
)
IAM_CREDENTIALS = Credentials(
    access_key="AKIDEXAMPLE",
    secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
)
EMPTY_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def signature(headers):
    return headers["authorization"].split("Signature=")[1]


class KnownVectorTests(unittest.TestCase):
    def test_get_object_signature(self):
        headers = sign_request(
            method="GET",
            url="https://examplebucket.s3.amazonaws.com/test.txt",
            body=b"",
            headers={
                "host": "examplebucket.s3.amazonaws.com",
                "range": "bytes=0-9",
                "x-amz-content-sha256": EMPTY_HASH,
            },
            credentials=S3_CREDENTIALS,
            service="s3",
            region="us-east-1",
            timestamp="20130524T000000Z",
        )
        self.assertEqual(
            signature(headers),
            "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41",
        )

    def test_get_bucket_lifecycle_signature(self):
        headers = sign_request(
            method="GET",
            url="https://examplebucket.s3.amazonaws.com/?lifecycle",
            body=b"",
            headers={"host": "examplebucket.s3.amazonaws.com", "x-amz-content-sha256": EMPTY_HASH},
            credentials=S3_CREDENTIALS,
            service="s3",
            region="us-east-1",
            timestamp="20130524T000000Z",
        )
        self.assertEqual(
            signature(headers),
            "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543",
        )

    def test_empty_post(self):
        headers = sign_request(
            method="POST",
            url="https://example.amazonaws.com/",
            body=b"",
            headers={"host": "example.amazonaws.com"},
            credentials=IAM_CREDENTIALS,
            service="service",
            region="us-east-1",
            timestamp="20150830T123600Z",
        )
        self.assertEqual(
            signature(headers),
            "5da7c1a2acd57cee7505fc6676e4e544621c30862966e37dddb68e92efbe5d6b",
        )

    def test_query_parameters_are_sorted(self):
        headers = sign_request(
            method="GET",
            url="https://example.amazonaws.com/?Param2=value2&Param1=value1",
            body=b"",
            headers={"host": "example.amazonaws.com"},
            credentials=IAM_CREDENTIALS,
            service="service",
            region="us-east-1",
            timestamp="20150830T123600Z",
        )
        self.assertEqual(
            signature(headers),
            "b97d918cfa904a5beff61c982a1b6f458b799221646efd99d3219ec94cdf2500",
        )


class BehaviourTests(unittest.TestCase):
    def test_signing_key_matches_aws_derivation(self):
        # The derived key from AWS's "Examples of how to derive a signing key".
        key = signing_key(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "20120215", "us-east-1", "iam"
        )
        self.assertEqual(
            key.hex(),
            "f4780e2d9f65fa895f9c67b32ce1baf0b0d8a43505a000a1a9e090d414db404d",
        )

    def test_host_defaults_from_url(self):
        headers = sign_request(
            method="POST",
            url="https://bedrock-runtime.us-west-2.amazonaws.com/model/x/converse",
            body=b"{}",
            headers={"content-type": "application/json"},
            credentials=S3_CREDENTIALS,
            service="bedrock",
            region="us-west-2",
            timestamp="20150830T123600Z",
        )
        self.assertIn("host;", headers["authorization"])
        self.assertNotIn("host", headers)  # returned as-is, not injected

    def test_session_token_is_signed_and_sent(self):
        credentials = Credentials("AK", "SK", session_token="token")
        headers = sign_request(
            method="POST",
            url="https://example.amazonaws.com/",
            body=b"",
            headers={"host": "example.amazonaws.com"},
            credentials=credentials,
            service="service",
            region="us-east-1",
            timestamp="20150830T123600Z",
        )
        self.assertEqual(headers["x-amz-security-token"], "token")
        self.assertIn("x-amz-security-token", headers["authorization"])

    def test_body_changes_the_signature(self):
        first = sign_request(
            method="POST",
            url="https://example.amazonaws.com/",
            body=b'{"a": 1}',
            headers={"host": "example.amazonaws.com"},
            credentials=S3_CREDENTIALS,
            service="service",
            region="us-east-1",
            timestamp="20150830T123600Z",
        )
        second = sign_request(
            method="POST",
            url="https://example.amazonaws.com/",
            body=b'{"a":2}',
            headers={"host": "example.amazonaws.com"},
            credentials=S3_CREDENTIALS,
            service="service",
            region="us-east-1",
            timestamp="20150830T123600Z",
        )
        self.assertNotEqual(signature(first), signature(second))

    def test_timestamp_format(self):
        self.assertRegex(now(), r"^\d{8}T\d{6}Z$")


if __name__ == "__main__":
    unittest.main()
