# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for sandbox_shared.py download_file method."""

import asyncio
import tarfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from verl.recipe.real.tools.sandbox_shared import SandboxSession


class TestDownloadFile:
    """Test cases for SandboxSession.download_file method."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock Environment object."""
        return MagicMock()

    @pytest.fixture
    def sandbox_session(self, mock_env):
        """Create a SandboxSession instance with mocked env."""
        return SandboxSession(env=mock_env, session_id="test-session-123")

    @pytest.fixture
    def temp_local_path(self, tmp_path):
        """Create a temporary local path for downloads."""
        return tmp_path / "downloads"

    @pytest.fixture
    def sample_tar_content(self):
        """Create sample tar.gz content with a test file inside."""
        tar_buffer = BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            # Add a simple text file to the archive
            data = b"Hello, World!"
            info = tarfile.TarInfo(name="test_file.txt")
            info.size = len(data)
            tar.addfile(info, BytesIO(data))
        return tar_buffer.getvalue()

    @pytest.mark.asyncio
    async def test_download_file_success_single_bos_url(
        self, sandbox_session, temp_local_path, sample_tar_content
    ):
        """Test successful download with single bos_url response."""
        with patch("aiohttp.ClientSession") as mock_client_session_cls:
            # Setup mock HTTP response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=sample_tar_content)
            mock_response.raise_for_status = MagicMock()

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_client_session_cls.return_value = mock_session

            # Setup mock session_download response
            sandbox_session.env.session_download = AsyncMock(
                return_value={
                    "bos_url": "https://example.com/download.tar.gz",
                    "file_name": "test_download",
                }
            )

            # Execute download
            result = await sandbox_session.download_file(
                remote_path="/mnt/user-data/outputs/test.txt",
                local_path=temp_local_path,
            )

            # Verify
            assert result is True
            sandbox_session.env.session_download.assert_called_once_with(
                session_id="test-session-123",
                paths=["/mnt/user-data/outputs/test.txt"],
                auto_extract=False,
            )

            # Verify extracted file exists
            extracted_file = temp_local_path / "test_file.txt"
            assert extracted_file.exists()
            assert extracted_file.read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_download_file_success_multiple_bos_urls(
        self, sandbox_session, temp_local_path, sample_tar_content
    ):
        """Test successful download with multiple bos_urls response."""
        with patch("aiohttp.ClientSession") as mock_client_session_cls:
            # Setup mock HTTP response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=sample_tar_content)
            mock_response.raise_for_status = MagicMock()

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_client_session_cls.return_value = mock_session

            # Setup mock session_download response with multiple URLs
            sandbox_session.env.session_download = AsyncMock(
                return_value={
                    "bos_urls": [
                        "https://example.com/file1.tar.gz",
                        "https://example.com/file2.tar.gz",
                    ],
                    "files": ["file1", "file2"],
                }
            )

            # Execute download
            result = await sandbox_session.download_file(
                remote_path="/mnt/user-data/outputs/",
                local_path=temp_local_path,
            )

            # Verify
            assert result is True
            assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_download_file_no_bos_urls(self, sandbox_session, temp_local_path):
        """Test download when no bos_urls are returned."""
        # Setup mock session_download response with no URLs
        sandbox_session.env.session_download = AsyncMock(return_value={})

        # Execute download
        result = await sandbox_session.download_file(
            remote_path="/mnt/user-data/outputs/test.txt",
            local_path=temp_local_path,
        )

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_download_file_no_bos_urls_none_response(
        self, sandbox_session, temp_local_path
    ):
        """Test download when session_download returns None."""
        # Setup mock session_download response as None
        sandbox_session.env.session_download = AsyncMock(return_value=None)

        # Execute download
        result = await sandbox_session.download_file(
            remote_path="/mnt/user-data/outputs/test.txt",
            local_path=temp_local_path,
        )

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_download_file_retry_on_http_error(
        self, sandbox_session, temp_local_path, sample_tar_content
    ):
        """Test retry mechanism when HTTP request fails."""
        call_count = 0

        def create_mock_response():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # First two attempts fail
                mock_resp = AsyncMock()
                mock_resp.raise_for_status = MagicMock(
                    side_effect=Exception("HTTP Error 500")
                )
                return mock_resp
            else:
                # Third attempt succeeds
                mock_resp = AsyncMock()
                mock_resp.status = 200
                mock_resp.read = AsyncMock(return_value=sample_tar_content)
                mock_resp.raise_for_status = MagicMock()
                return mock_resp

        with patch("aiohttp.ClientSession") as mock_client_session_cls:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(side_effect=create_mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_client_session_cls.return_value = mock_session

            sandbox_session.env.session_download = AsyncMock(
                return_value={
                    "bos_url": "https://example.com/download.tar.gz",
                    "file_name": "test",
                }
            )

            # Execute download
            result = await sandbox_session.download_file(
                remote_path="/mnt/user-data/outputs/test.txt",
                local_path=temp_local_path,
            )

            # Verify retry succeeded
            assert result is True
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_download_file_retry_on_session_download_error(
        self, sandbox_session, temp_local_path, sample_tar_content
    ):
        """Test retry mechanism when session_download fails."""
        call_count = 0

        async def mock_session_download(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Session download error")
            return {
                "bos_url": "https://example.com/download.tar.gz",
                "file_name": "test",
            }

        with patch("aiohttp.ClientSession") as mock_client_session_cls:
            # Setup mock HTTP response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=sample_tar_content)
            mock_response.raise_for_status = MagicMock()

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_client_session_cls.return_value = mock_session

            sandbox_session.env.session_download = mock_session_download

            # Execute download
            result = await sandbox_session.download_file(
                remote_path="/mnt/user-data/outputs/test.txt",
                local_path=temp_local_path,
            )

            # Verify retry succeeded
            assert result is True
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_download_file_fail_after_max_retries(
        self, sandbox_session, temp_local_path
    ):
        """Test download fails after maximum retries."""
        sandbox_session.env.session_download = AsyncMock(
            side_effect=Exception("Persistent error")
        )

        # Execute download
        result = await sandbox_session.download_file(
            remote_path="/mnt/user-data/outputs/test.txt",
            local_path=temp_local_path,
        )

        # Verify
        assert result is False
        assert sandbox_session.env.session_download.call_count == 3

    @pytest.mark.asyncio
    async def test_download_file_creates_local_directory(
        self, sandbox_session, tmp_path, sample_tar_content
    ):
        """Test that local directory is created if it doesn't exist."""
        non_existent_path = tmp_path / "new_dir" / "subdir"

        with patch("aiohttp.ClientSession") as mock_client_session_cls:
            # Setup mock HTTP response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=sample_tar_content)
            mock_response.raise_for_status = MagicMock()

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_client_session_cls.return_value = mock_session

            sandbox_session.env.session_download = AsyncMock(
                return_value={
                    "bos_url": "https://example.com/download.tar.gz",
                    "file_name": "test",
                }
            )

            # Execute download
            result = await sandbox_session.download_file(
                remote_path="/mnt/user-data/outputs/test.txt",
                local_path=non_existent_path,
            )

            # Verify directory was created
            assert result is True
            assert non_existent_path.exists()
            assert non_existent_path.is_dir()

    @pytest.mark.asyncio
    async def test_download_file_appends_tar_gz_extension(
        self, sandbox_session, temp_local_path, sample_tar_content
    ):
        """Test that .tar.gz extension is appended if not present in file name."""
        with patch("aiohttp.ClientSession") as mock_client_session_cls:
            # Setup mock HTTP response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=sample_tar_content)
            mock_response.raise_for_status = MagicMock()

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_client_session_cls.return_value = mock_session

            sandbox_session.env.session_download = AsyncMock(
                return_value={
                    "bos_url": "https://example.com/download",
                    "file_name": "my_file",  # No .tar.gz extension
                }
            )

            # Execute download
            result = await sandbox_session.download_file(
                remote_path="/mnt/user-data/outputs/test.txt",
                local_path=temp_local_path,
            )

            # Verify
            assert result is True
            # The tar file should be cleaned up after extraction
            tar_file = temp_local_path / "my_file.tar.gz"
            assert not tar_file.exists()  # Should be deleted after extraction

    @pytest.mark.asyncio
    async def test_download_file_default_file_name(
        self, sandbox_session, temp_local_path, sample_tar_content
    ):
        """Test default file name when file_name is not provided."""
        with patch("aiohttp.ClientSession") as mock_client_session_cls:
            # Setup mock HTTP response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=sample_tar_content)
            mock_response.raise_for_status = MagicMock()

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_client_session_cls.return_value = mock_session

            sandbox_session.env.session_download = AsyncMock(
                return_value={
                    "bos_url": "https://example.com/download.tar.gz",
                    # No file_name provided
                }
            )

            # Execute download
            result = await sandbox_session.download_file(
                remote_path="/mnt/user-data/outputs/test.txt",
                local_path=temp_local_path,
            )

            # Verify
            assert result is True

    @pytest.mark.asyncio
    async def test_download_file_multiple_files_missing_names(
        self, sandbox_session, temp_local_path, sample_tar_content
    ):
        """Test download with multiple URLs but some file names missing."""
        with patch("aiohttp.ClientSession") as mock_client_session_cls:
            # Setup mock HTTP response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=sample_tar_content)
            mock_response.raise_for_status = MagicMock()

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_client_session_cls.return_value = mock_session

            # 3 URLs but only 1 file name provided
            sandbox_session.env.session_download = AsyncMock(
                return_value={
                    "bos_urls": [
                        "https://example.com/file1.tar.gz",
                        "https://example.com/file2.tar.gz",
                        "https://example.com/file3.tar.gz",
                    ],
                    "files": ["file1"],  # Only one file name
                }
            )

            # Execute download
            result = await sandbox_session.download_file(
                remote_path="/mnt/user-data/outputs/",
                local_path=temp_local_path,
            )

            # Verify
            assert result is True
            assert mock_session.get.call_count == 3
