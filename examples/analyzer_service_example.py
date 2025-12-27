"""Small example showing `AnalyzerService` usage with a dummy provider.

Run directly to see output:
    python examples/analyzer_service_example.py
"""
from ai_image_analyzer.services.analyzer import AnalyzerService


class DummyProvider:
    def call_text(self, cfg, text, system_prompt=None, quiet=False):
        return f"echo: {text}", {"total_cost": 0.0}

    def call_image(self, cfg, jpeg_bytes, system_prompt=None, user_text=None, quiet=False, image_meta=None):
        return "image-analysed", {"total_cost": 0.0}

    def check_balance(self, cfg, quiet=False):
        return {"balance": 123.45}


def main():
    provider = DummyProvider()
    svc = AnalyzerService(provider)

    text_res, usage = svc.analyze_text(None, "hello world")
    print("Text result:", text_res, "usage:", usage)

    img_res, usage = svc.analyze_image(None, b"\xFF\xD8\xFF")
    print("Image result:", img_res, "usage:", usage)


if __name__ == "__main__":
    main()
