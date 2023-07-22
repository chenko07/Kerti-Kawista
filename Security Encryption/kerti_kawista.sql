-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Waktu pembuatan: 23 Jul 2023 pada 00.20
-- Versi server: 10.4.28-MariaDB
-- Versi PHP: 8.2.4

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `kerti_kawista`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `admin`
--

CREATE TABLE `admin` (
  `id` int(11) NOT NULL,
  `name_admin` varchar(255) NOT NULL,
  `username` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `admin`
--

INSERT INTO `admin` (`id`, `name_admin`, `username`, `password`) VALUES
(1, 'rendie', 'admin', 'admin');

-- --------------------------------------------------------

--
-- Struktur dari tabel `admin_encrypted`
--

CREATE TABLE `admin_encrypted` (
  `id` int(11) NOT NULL,
  `name_admin` varchar(255) NOT NULL,
  `username` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `admin_encrypted`
--

INSERT INTO `admin_encrypted` (`id`, `name_admin`, `username`, `password`) VALUES
(1, 'rendie', '057ad3efd7cbba11180940f03f616eb1', 'e8bb0cf90b7fc80f792205ecd1c0966a');

-- --------------------------------------------------------

--
-- Struktur dari tabel `user`
--

CREATE TABLE `user` (
  `id` int(11) NOT NULL,
  `name_user` varchar(255) NOT NULL,
  `username` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `role_pekerjaan` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `user`
--

INSERT INTO `user` (`id`, `name_user`, `username`, `password`, `role_pekerjaan`) VALUES
(1, 'Rendie Abdi Saputra', 'rendie', 'rendie12345', 'pekerja buruh biasa'),
(2, 'Alvian Putra Hardiadi', 'alvian', 'alvian12345', 'safety officer'),
(3, 'Marchel Andrian Shevchenko', 'chenko', 'chenko12345', 'manajer'),
(4, 'Nicholas Dwinata', 'nicholas', 'nicholas12345', 'pengunjung');

-- --------------------------------------------------------

--
-- Struktur dari tabel `user_encrypted`
--

CREATE TABLE `user_encrypted` (
  `id` int(11) NOT NULL,
  `name_user` varchar(255) NOT NULL,
  `username` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `role_pekerjaan` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `user_encrypted`
--

INSERT INTO `user_encrypted` (`id`, `name_user`, `username`, `password`, `role_pekerjaan`) VALUES
(1, 'Rendie Abdi Saputra', 'f2f1e76b78c7f3ca4cce5cbafac7f65e', 'c27be182c040bb650df78868fbb936db', 'ca11b198ec62a08694be7c4d34686801'),
(2, 'Alvian Putra Hardiadi', '00592237610f73ff4edf7c66d47b7324', '06622a862bc850fc38de0257451860d9', '182a7722c75e7c894fd91690f3b7a025'),
(3, 'Marchel Andrian Shevchenko', '69df996c3b27a355278d4000c5545fd5', '553ef9726ad2875674ef2a3edd84c15f', '81be7a5b39c48ace313eaa62a9b6a656'),
(4, 'Nicholas Dwinata', 'c66205f17318c6991eaff775124445cb', 'b831ff28a83d419c84b3fd93f0b27dcf', '0155bfa7db8bc9c8a639b52ea527f882');

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `admin`
--
ALTER TABLE `admin`
  ADD PRIMARY KEY (`id`);

--
-- Indeks untuk tabel `admin_encrypted`
--
ALTER TABLE `admin_encrypted`
  ADD PRIMARY KEY (`id`);

--
-- Indeks untuk tabel `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`id`);

--
-- Indeks untuk tabel `user_encrypted`
--
ALTER TABLE `user_encrypted`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `admin`
--
ALTER TABLE `admin`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT untuk tabel `admin_encrypted`
--
ALTER TABLE `admin_encrypted`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT untuk tabel `user`
--
ALTER TABLE `user`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT untuk tabel `user_encrypted`
--
ALTER TABLE `user_encrypted`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
