{
  description = "splatrs: 3D Gaussian splatting trainer and renderer, CPU reference implementation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        # The lock file must exist: run `cargo generate-lockfile` once in the dev shell and
        # commit the result before `nix build`.
        splatrs = pkgs.rustPlatform.buildRustPackage {
          pname = "splatrs";
          version = "0.1.0";
          src = pkgs.lib.cleanSourceWith {
            src = pkgs.lib.cleanSource ./.;
            filter = path: _type: baseNameOf path != "target";
          };
          cargoLock.lockFile = ./Cargo.lock;
          meta = {
            description = "3D Gaussian splatting trainer and renderer";
            license = pkgs.lib.licenses.agpl3Plus;
            mainProgram = "splatrs";
          };
        };
      in
      {
        packages.default = splatrs;
        packages.splatrs = splatrs;

        apps.default = {
          type = "app";
          program = "${splatrs}/bin/splatrs";
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rustc
            clippy
            rustfmt
            rust-analyzer
            # Structure from motion for the camera poses and the initial point cloud.
            colmap
          ];
          RUST_BACKTRACE = "1";
        };
      });
}