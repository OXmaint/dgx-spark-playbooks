/*
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/
"use client";

import { useState, useEffect, useRef } from 'react';

export default function OrganizationSelector() {
  const [organizations, setOrganizations] = useState<string[]>([]);
  const [selectedOrg, setSelectedOrg] = useState<string>("Oxmaint");
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Load organizations from JSON file
    const loadOrganizations = async () => {
      try {
        const response = await fetch('/organizations.json');
        const data = await response.json();
        setOrganizations(data.organizations || []);

        // Load saved organization from localStorage or default to first
        const savedOrg = localStorage.getItem('selectedOrganization');
        if (savedOrg && data.organizations.includes(savedOrg)) {
          setSelectedOrg(savedOrg);
        } else if (data.organizations.length > 0) {
          setSelectedOrg(data.organizations[0]);
        }
      } catch (error) {
        console.error('Error loading organizations:', error);
        setOrganizations(['Oxmaint']); // Fallback to default
      }
    };

    loadOrganizations();
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleOrgSelect = (org: string) => {
    setSelectedOrg(org);
    localStorage.setItem('selectedOrganization', org);
    setIsOpen(false);
  };

  return (
    <div className="fixed top-4 right-20 z-50" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 rounded-full bg-white dark:bg-gray-800 shadow-lg hover:shadow-xl transition-all duration-200 z-50"
        aria-label="Select organization"
      >
        <svg
          className="w-6 h-6 text-gray-700 dark:text-gray-300"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"
          />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute top-12 right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
          {organizations.map((org) => (
            <button
              key={org}
              onClick={() => handleOrgSelect(org)}
              className={`w-full text-left px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-150 ${
                selectedOrg === org
                  ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 font-medium'
                  : 'text-gray-900 dark:text-white'
              }`}
            >
              <div className="flex items-center justify-between">
                <span>{org}</span>
                {selectedOrg === org && (
                  <svg
                    className="w-5 h-5"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                )}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function useSelectedOrganization() {
  const [selectedOrg, setSelectedOrg] = useState<string>("Oxmaint");

  useEffect(() => {
    const loadSelectedOrg = () => {
      const savedOrg = localStorage.getItem('selectedOrganization');
      if (savedOrg) {
        setSelectedOrg(savedOrg);
      }
    };

    loadSelectedOrg();

    // Listen for changes to localStorage
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'selectedOrganization' && e.newValue) {
        setSelectedOrg(e.newValue);
      }
    };

    window.addEventListener('storage', handleStorageChange);

    // Also listen for changes in the same tab
    const intervalId = setInterval(loadSelectedOrg, 500);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      clearInterval(intervalId);
    };
  }, []);

  return selectedOrg;
}
