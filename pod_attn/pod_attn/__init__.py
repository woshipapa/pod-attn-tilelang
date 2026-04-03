__version__ = "1.0.0"

try:
    from pod_attn.fused_attn_interface import (  # noqa: F401
        true_fused_attn_with_kvcache,
    )
except Exception:
    pass

try:
    from pod_attn.flash_attn_interface import (  # noqa: F401
        flash_attn_with_kvcache,
    )
except Exception:
    pass

try:
    from pod_attn.tilelang_fused_attn import (  # noqa: F401
        true_fused_attn_with_kvcache_tilelang,
        smoke_test_tilelang_true_fused,
        get_tilelang_fused_launch_plan,
    )
except Exception:
    # TileLang is optional for this package.
    pass
