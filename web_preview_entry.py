"""Separate preview entry: never replaces the stable desktop package."""
import json
import sys


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        try:
            from core.package_health import run_package_self_check
            result = run_package_self_check()
            from ui.web_window import ASSET_DIR, ASSET_NAMES
            result['web_assets_present'] = all((ASSET_DIR/name).is_file() for name in ASSET_NAMES)
            if not result['web_assets_present']:
                result['status'] = 'failed'
        except Exception:
            result = {'status':'failed','mode':'offline_package_self_check'}
        print(json.dumps(result,ensure_ascii=False))
        raise SystemExit(0 if result.get('status') == 'ok' else 1)
    if sys.argv[1:] == ['--offline-ui-check']:
        from tools.webui_offline_smoke import main as smoke
        raise SystemExit(smoke())
    import main
    sys.argv.append('--web-ui')
    main.main()
