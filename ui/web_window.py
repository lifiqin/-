"""Opt-in, offline-first JavaScript desktop shell hosted by Qt WebEngine."""
from pathlib import Path

from PySide6.QtCore import QUrl, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings, QWebEngineUrlRequestInterceptor
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QMainWindow, QMessageBox

from core.config import APP_NAME, APP_VERSION
from ui.web_bridge import WebBridge


ASSET_DIR = Path(__file__).resolve().parent / 'web'
ASSET_NAMES = frozenset({'index.html', 'app.js', 'view_model.js', 'styles.css'})


def _allowed_request(url, asset_dir=ASSET_DIR):
    """Only bundled assets, Qt's WebChannel script and JPEG previews are loadable."""
    scheme = url.scheme().lower()
    if scheme == 'qrc':
        return url.path() == '/qtwebchannel/qwebchannel.js'
    if scheme == 'data':
        return url.toString().startswith('data:image/jpeg;base64,')
    if scheme != 'file' or url.host():
        return False
    try:
        path = Path(url.toLocalFile()).resolve(strict=True)
        root = Path(asset_dir).resolve(strict=True)
        return path.parent == root and path.name in ASSET_NAMES and path.is_file()
    except (OSError, ValueError, RuntimeError):
        return False


class _AssetInterceptor(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self, info):
        if not _allowed_request(info.requestUrl()):
            info.block(True)


class _RestrictedPage(QWebEnginePage):
    def acceptNavigationRequest(self, url, _kind, is_main_frame):
        if not is_main_frame or not _allowed_request(url):
            return False
        try:
            return Path(url.toLocalFile()).resolve(strict=True) == (ASSET_DIR / 'index.html').resolve(strict=True)
        except (OSError, ValueError, RuntimeError):
            return False

    def createWindow(self, _kind):
        return None


class WebMainWindow(QMainWindow):
    """The classic window remains available for functions not yet migrated."""

    def __init__(self, *, output_root=None):
        super().__init__()
        if not all((ASSET_DIR / name).is_file() for name in ASSET_NAMES):
            raise FileNotFoundError('web UI assets are unavailable')
        self.setWindowTitle(f'{APP_NAME} v{APP_VERSION} · JavaScript 界面预览')
        self.setMinimumSize(1000, 700)
        self.resize(1440, 900)
        self._legacy = None
        self._closing_after_workers = False
        self.bridge = WebBridge(self, output_root=output_root)
        self.bridge.legacyRequested.connect(self._open_legacy)
        self.bridge.shutdownReady.connect(self._finish_close)
        # An unnamed profile is off-the-record and does not retain browser cookies.
        self.profile = QWebEngineProfile(self)
        self.interceptor = _AssetInterceptor(self.profile)
        self.profile.setUrlRequestInterceptor(self.interceptor)
        self.page = _RestrictedPage(self.profile, self)
        self.channel = QWebChannel(self.page)
        self.channel.registerObject('bridge', self.bridge)
        self.page.setWebChannel(self.channel)
        settings = self.page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, False)
        self.view = QWebEngineView(self)
        self.view.setPage(self.page)
        self.setCentralWidget(self.view)
        self.view.load(QUrl.fromLocalFile(str((ASSET_DIR / 'index.html').resolve())))

    @Slot()
    def _open_legacy(self):
        if self.bridge._busy():
            return
        from ui.main_window import MainWindow
        self._legacy = MainWindow()
        self._legacy.show()
        self.close()

    @Slot()
    def _finish_close(self):
        if self._closing_after_workers:
            self.close()

    def closeEvent(self, event):
        if self._closing_after_workers:
            if self.bridge.shutdown():
                event.accept()
            else:
                event.ignore()
            return
        if self.bridge._busy():
            if QMessageBox.question(self, '任务进行中', '关闭会请求停止分析，并等待后台任务结束。继续吗？',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                event.ignore()
                return
            self._closing_after_workers = True
            if not self.bridge.shutdown():
                event.ignore()
                return
        else:
            self.bridge.shutdown()
        event.accept()
