{ ... }:
{
  perSystem =
    {
      config,
      pkgs,
      ...
    }:
    let
      python = pkgs.python314;
      uv = pkgs.uv;

      commonPackages = [
        uv
        python
        pkgs.cacert
        pkgs.nil
        pkgs.just
      ];

      uvSync = ''
        if [ -s coding/python/hypothesis/pyproject.toml ]; then
          echo "Synchronizing Python dependencies with uv..."
          uv sync --directory coding/python/hypothesis --all-groups
          export PATH="$PWD/coding/python/hypothesis/.venv/bin:$PATH"
        fi

        ${config.pre-commit.installationScript}
      '';

    in
    {
      devShells.default = pkgs.mkShell {
        inputsFrom = [ config.treefmt.build.devShell ];
        name = "master-thesis";
        packages = commonPackages;
        shellHook = uvSync;

        env = {
          UV_PYTHON_PREFERENCE = "only-managed";
          PYTHONDONTWRITEBYTECODE = "1";
          SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
        };
      };
    };
}
